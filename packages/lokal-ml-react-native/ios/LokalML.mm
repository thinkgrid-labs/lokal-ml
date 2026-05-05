/**
 * LokalML iOS JSI HostObject bridge.
 *
 * Registers `LokalMLNative` on the JS runtime global and exposes:
 *
 *   checkRequirements(modelId: string): boolean
 *   downloadModel(modelId: string, requireWifi: boolean,
 *                 onProgress?: (p: number) => void): Promise<void>
 *   isModelCached(modelId: string): boolean
 *   deleteModel(modelId: string): void
 *   initEngine(config: { model: string }): Promise<number>
 *   chat(handle: number,
 *        opts: { prompt: string, onToken?: (t: string) => void }): Promise<ChatResponse>
 *   disposeEngine(handle: number): void
 *
 * ChatResponse = { text: string, promptTokens: number,
 *                  generatedTokens: number, inferenceMs: number }
 *
 * Threading model
 * ───────────────
 * JSI Runtime calls arrive on the JS thread.  Blocking operations (initEngine,
 * chat) are dispatched to a background GCD queue.  Results are delivered back
 * to the JS thread via callInvoker->invokeAsync so that JSI objects are only
 * ever touched from the correct thread.
 */

#import <React/RCTBridgeModule.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>
#import <dispatch/dispatch.h>
#import "lokal-ml.h"

// CallInvoker is transitively included by RCTTurboModule.h on RN >= 0.71.
// If your project uses an older layout, add:
//   #include <ReactCommon/CallInvoker.h>
#include <memory>
#include <mutex>
#include <string>

using namespace facebook::jsi;
namespace react = facebook::react;

// ─── Path helper ─────────────────────────────────────────────────────────────

static std::string modelCachePath(const std::string& modelId) {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(
      NSCachesDirectory, NSUserDomainMask, YES);
  NSString *dir  = [paths firstObject];
  NSString *file = [[NSString stringWithUTF8String:modelId.c_str()]
                    stringByAppendingPathExtension:@"gguf"];
  return std::string([[dir stringByAppendingPathComponent:file] UTF8String]);
}

// ─── Promise helper ──────────────────────────────────────────────────────────
//
// Creates a JS Promise and synchronously calls `body(resolve, reject)` inside
// the executor so the caller can start async work before returning the Promise.

static Value makePromise(
    Runtime& rt,
    std::function<void(std::shared_ptr<Function> resolve,
                       std::shared_ptr<Function> reject)> body) {
  return rt.global()
           .getPropertyAsFunction(rt, "Promise")
           .callAsConstructor(
               rt,
               Function::createFromHostFunction(
                   rt, PropNameID::forAscii(rt, ""), 2,
                   [body = std::move(body)](
                       Runtime& rt, const Value&,
                       const Value* args, size_t) mutable -> Value {
                     auto resolve = std::make_shared<Function>(
                         args[0].asObject(rt).asFunction(rt));
                     auto reject  = std::make_shared<Function>(
                         args[1].asObject(rt).asFunction(rt));
                     body(std::move(resolve), std::move(reject));
                     return Value::undefined();
                   }));
}

// Reject a promise with a plain Error object.
static void rejectWithMessage(
    Runtime* rt,
    const std::shared_ptr<Function>& reject,
    const std::string& msg) {
  auto err = rt->global()
               .getPropertyAsFunction(*rt, "Error")
               .callAsConstructor(*rt, String::createFromUtf8(*rt, msg));
  reject->call(*rt, err);
}

// ─── Chat callback context ────────────────────────────────────────────────────
//
// Heap-allocated; passed as user_data through both C callbacks.
// Freed inside chatOnComplete (which is always called exactly once).

struct ChatCtx {
  std::shared_ptr<react::CallInvoker> invoker;
  Runtime*                            rt;
  std::shared_ptr<Function>           onTokenFn;  // nullable
  std::shared_ptr<Function>           resolve;
  std::shared_ptr<Function>           reject;

  std::mutex   textMu;
  std::string  fullText;
};

static void chatOnToken(const char* token, void* ud) {
  auto* ctx = static_cast<ChatCtx*>(ud);
  std::string tok(token ? token : "");

  {
    std::lock_guard<std::mutex> lock(ctx->textMu);
    ctx->fullText += tok;
  }

  if (ctx->onTokenFn) {
    auto rt  = ctx->rt;
    auto fn  = ctx->onTokenFn;
    ctx->invoker->invokeAsync([rt, tok = std::move(tok), fn]() {
      fn->call(*rt, String::createFromUtf8(*rt, tok));
    });
  }
}

static void chatOnComplete(uint32_t genTokens, uint64_t ms, void* ud) {
  auto* ctx = static_cast<ChatCtx*>(ud);

  // Copy everything we need out of ctx before deleting it.
  auto invoker = ctx->invoker;
  auto rt      = ctx->rt;
  auto resolve = ctx->resolve;
  std::string full;
  {
    std::lock_guard<std::mutex> lock(ctx->textMu);
    full = ctx->fullText;
  }
  delete ctx;

  invoker->invokeAsync(
      [rt, resolve, full = std::move(full), genTokens, ms]() {
        auto resp = Object(*rt);
        resp.setProperty(*rt, "text",
                         String::createFromUtf8(*rt, full));
        resp.setProperty(*rt, "promptTokens",    Value(0.0));
        resp.setProperty(*rt, "generatedTokens", Value((double)genTokens));
        resp.setProperty(*rt, "inferenceMs",     Value((double)ms));
        resolve->call(*rt, resp);
      });
}

// ─── Download callback context ────────────────────────────────────────────────

struct DownloadCtx {
  std::shared_ptr<react::CallInvoker> invoker;
  Runtime*                            rt;
  std::shared_ptr<Function>           onProgressFn;  // nullable
  std::shared_ptr<Function>           resolve;
  std::shared_ptr<Function>           reject;
};

static void downloadOnProgress(float p, void* ud) {
  auto* ctx = static_cast<DownloadCtx*>(ud);
  if (!ctx->onProgressFn) return;

  auto rt = ctx->rt;
  auto fn = ctx->onProgressFn;
  ctx->invoker->invokeAsync([rt, p, fn]() {
    fn->call(*rt, Value((double)p));
  });
}

static void downloadOnComplete(bool success, void* ud) {
  auto* ctx = static_cast<DownloadCtx*>(ud);

  auto invoker = ctx->invoker;
  auto rt      = ctx->rt;
  auto resolve = ctx->resolve;
  auto reject  = ctx->reject;
  delete ctx;

  invoker->invokeAsync([rt, resolve, reject, success]() {
    if (success) {
      resolve->call(*rt, Value::undefined());
    } else {
      rejectWithMessage(rt, reject, "Download failed");
    }
  });
}

// ─── JSI HostObject ──────────────────────────────────────────────────────────

class LokalMLHostObject : public HostObject {
  std::shared_ptr<react::CallInvoker> invoker_;
  Runtime*                            rt_;

public:
  LokalMLHostObject(std::shared_ptr<react::CallInvoker> invoker, Runtime* rt)
      : invoker_(std::move(invoker)), rt_(rt) {}

  Value get(Runtime& rt, const PropNameID& name) override {
    auto n = name.utf8(rt);

    // ── checkRequirements ──────────────────────────────────────────────────
    if (n == "checkRequirements") {
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 1,
          [](Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1 || !args[0].isString()) return Value(false);
            return Value(lokal_check_requirements(
                args[0].asString(rt).utf8(rt).c_str()));
          });
    }

    // ── downloadModel ──────────────────────────────────────────────────────
    if (n == "downloadModel") {
      auto invoker = invoker_;
      auto rt_ptr  = rt_;
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 3,
          [invoker, rt_ptr](
              Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1 || !args[0].isString()) return Value::undefined();

            std::string modelId  = args[0].asString(rt).utf8(rt);
            bool requireWifi = (c >= 2 && args[1].isBool()) ? args[1].asBool() : true;

            std::shared_ptr<Function> onProgressFn;
            if (c >= 3 && args[2].isObject() &&
                args[2].asObject(rt).isFunction(rt)) {
              onProgressFn = std::make_shared<Function>(
                  args[2].asObject(rt).asFunction(rt));
            }

            return makePromise(rt, [=](auto resolve, auto reject) {
              auto* ctx = new DownloadCtx{
                  invoker, rt_ptr, onProgressFn, resolve, reject};
              lokal_download_model(
                  modelId.c_str(),
                  requireWifi,
                  onProgressFn ? downloadOnProgress : nullptr,
                  ctx,
                  downloadOnComplete,
                  ctx);
            });
          });
    }

    // ── isModelCached ──────────────────────────────────────────────────────
    if (n == "isModelCached") {
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 1,
          [](Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1 || !args[0].isString()) return Value(false);
            return Value(lokal_is_model_cached(
                args[0].asString(rt).utf8(rt).c_str()));
          });
    }

    // ── deleteModel ────────────────────────────────────────────────────────
    if (n == "deleteModel") {
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 1,
          [](Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1 || !args[0].isString()) return Value::undefined();
            lokal_delete_model(args[0].asString(rt).utf8(rt).c_str());
            return Value::undefined();
          });
    }

    // ── initEngine ─────────────────────────────────────────────────────────
    // Loading a model can take several seconds (memory mapping + warmup).
    // We dispatch to a background queue and resolve the Promise from the
    // JS thread via callInvoker so the JS thread is never blocked.
    if (n == "initEngine") {
      auto invoker = invoker_;
      auto rt_ptr  = rt_;
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 1,
          [invoker, rt_ptr](
              Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1 || !args[0].isObject()) return Value(0.0);

            std::string modelId =
                args[0].asObject(rt).getProperty(rt, "model")
                    .asString(rt).utf8(rt);
            // Support custom GGUF paths: if model starts with '/' treat it as
            // an absolute path the caller manages; otherwise resolve via cache.
            std::string modelPath = (!modelId.empty() && modelId[0] == '/')
                ? modelId
                : modelCachePath(modelId);

            return makePromise(rt, [=](auto resolve, auto reject) {
              dispatch_async(
                  dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                    uint32_t handle = lokal_init_engine(modelPath.c_str());
                    invoker->invokeAsync([rt_ptr, resolve, reject, handle]() {
                      if (handle == 0) {
                        rejectWithMessage(rt_ptr, reject,
                                          "Failed to load model");
                      } else {
                        resolve->call(*rt_ptr, Value((double)handle));
                      }
                    });
                  });
            });
          });
    }

    // ── chat ───────────────────────────────────────────────────────────────
    // Inference is dispatched to a background GCD queue.  Each token fires
    // chatOnToken which uses callInvoker->invokeAsync to call onToken on the
    // JS thread.  When inference finishes, chatOnComplete resolves the Promise.
    if (n == "chat") {
      auto invoker = invoker_;
      auto rt_ptr  = rt_;
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 2,
          [invoker, rt_ptr](
              Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 2 || !args[0].isNumber() || !args[1].isObject()) {
              return Value::undefined();
            }
            uint32_t handle = (uint32_t)args[0].asNumber();
            auto opts        = args[1].asObject(rt);
            std::string prompt =
                opts.getProperty(rt, "prompt").asString(rt).utf8(rt);

            std::shared_ptr<Function> onTokenFn;
            auto onTokenProp = opts.getProperty(rt, "onToken");
            if (onTokenProp.isObject() &&
                onTokenProp.asObject(rt).isFunction(rt)) {
              onTokenFn = std::make_shared<Function>(
                  onTokenProp.asObject(rt).asFunction(rt));
            }

            return makePromise(rt, [=](auto resolve, auto reject) {
              auto* ctx  = new ChatCtx;
              ctx->invoker    = invoker;
              ctx->rt         = rt_ptr;
              ctx->onTokenFn  = onTokenFn;
              ctx->resolve    = resolve;
              ctx->reject     = reject;

              dispatch_async(
                  dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                    lokal_chat_stream(
                        handle,
                        prompt.c_str(),
                        chatOnToken,
                        ctx,
                        chatOnComplete,
                        ctx);
                  });
            });
          });
    }

    // ── disposeEngine ──────────────────────────────────────────────────────
    if (n == "disposeEngine") {
      return Function::createFromHostFunction(
          rt, PropNameID::forAscii(rt, n.c_str()), 1,
          [](Runtime& rt, const Value&, const Value* args, size_t c) -> Value {
            if (c < 1) return Value::undefined();
            lokal_dispose_engine((uint32_t)args[0].asNumber());
            return Value::undefined();
          });
    }

    return Value::undefined();
  }

  void set(Runtime&, const PropNameID&, const Value&) override {}
};

// ─── React Native Module Registration ────────────────────────────────────────

@interface LokalML : NSObject <RCTBridgeModule>
@end

@implementation LokalML

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup { return NO; }

- (void)setBridge:(RCTBridge *)bridge {
  // Capture the call invoker before jumping to the JS thread — it is safe to
  // read from any thread; only the invoker *callbacks* must run on the JS thread.
  auto callInvoker = bridge.jsCallInvoker;

  [bridge dispatchBlock:^{
    Runtime* rt = bridge.runtime;
    if (!rt || !callInvoker) return;

    auto hostObj = std::make_shared<LokalMLHostObject>(callInvoker, rt);
    rt->global().setProperty(
        *rt, "LokalMLNative",
        Object::createFromHostObject(*rt, hostObj));
  } queue:RCTJSThread];
}

@end
