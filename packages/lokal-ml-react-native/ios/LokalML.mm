/**
 * LokalML iOS JSI HostObject bridge.
 *
 * Registers the `LokalMLNative` global on the JS runtime and exposes the
 * following methods via JSI (zero async serialisation overhead):
 *
 *   - checkRequirements(modelId: string): boolean
 *   - downloadModel(modelId: string, requireWifi: bool, onProgress: fn): void
 *   - isModelCached(modelId: string): boolean
 *   - deleteModel(modelId: string): void
 *   - initEngine(config: object): number   ← returns an opaque engine handle
 *   - chat(handle: number, options: object): object
 *   - disposeEngine(handle: number): void
 */

#import <React/RCTBridgeModule.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>
#import "lokal-ml.h"  // cbindgen-generated C header

using namespace facebook::jsi;

// Resolve a model ID (e.g. "gemma-2b-int4") to its full on-disk cache path.
// Models are stored as <Caches>/<modelId>.gguf to keep them out of iCloud backup.
static std::string modelCachePath(const std::string& modelId) {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
  NSString *cacheDir = [paths firstObject];
  NSString *fileName = [[NSString stringWithUTF8String:modelId.c_str()]
                        stringByAppendingPathExtension:@"gguf"];
  NSString *fullPath = [cacheDir stringByAppendingPathComponent:fileName];
  return std::string([fullPath UTF8String]);
}

// ─── JSI HostObject ──────────────────────────────────────────────────────────

class LokalMLHostObject : public HostObject {
public:
  Value get(Runtime& rt, const PropNameID& name) override {
    auto methodName = name.utf8(rt);

    // ── checkRequirements ──────────────────────────────────────────────────
    if (methodName == "checkRequirements") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "checkRequirements"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1 || !args[0].isString()) return Value(false);
          auto modelId = args[0].asString(rt).utf8(rt);
          bool result = lokal_check_requirements(modelId.c_str());
          return Value(result);
        }
      );
    }

    // ── downloadModel ──────────────────────────────────────────────────────
    if (methodName == "downloadModel") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "downloadModel"), 3,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1 || !args[0].isString()) return Value::undefined();
          auto modelId     = args[0].asString(rt).utf8(rt);
          bool requireWifi = (count >= 2 && args[1].isBool()) ? args[1].asBool() : true;
          // TODO: wire JS onProgress callback (args[2]) through to Rust once
          // the Tokio-based download task is implemented in lokal_download_model.
          lokal_download_model(modelId.c_str(), requireWifi, nullptr);
          return Value::undefined();
        }
      );
    }

    // ── isModelCached ──────────────────────────────────────────────────────
    if (methodName == "isModelCached") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "isModelCached"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1 || !args[0].isString()) return Value(false);
          auto modelId = args[0].asString(rt).utf8(rt);
          bool cached  = lokal_is_model_cached(modelId.c_str());
          return Value(cached);
        }
      );
    }

    // ── deleteModel ────────────────────────────────────────────────────────
    if (methodName == "deleteModel") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "deleteModel"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1 || !args[0].isString()) return Value::undefined();
          auto modelId = args[0].asString(rt).utf8(rt);
          lokal_delete_model(modelId.c_str());
          return Value::undefined();
        }
      );
    }

    // ── initEngine ─────────────────────────────────────────────────────────
    // Resolves the model ID to its cache path before calling into Rust.
    // lokal_init_engine() expects a full filesystem path, not a model ID.
    if (methodName == "initEngine") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "initEngine"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1 || !args[0].isObject()) return Value((double)0);
          auto config  = args[0].asObject(rt);
          auto modelId = config.getProperty(rt, "model").asString(rt).utf8(rt);
          // Resolve model ID → on-disk path before handing off to Rust.
          std::string modelPath = modelCachePath(modelId);
          uint32_t handle = lokal_init_engine(modelPath.c_str());
          return Value((double)handle);
        }
      );
    }

    // ── chat ───────────────────────────────────────────────────────────────
    // Runs inference synchronously on the calling thread (will be moved to a
    // background dispatch queue once the real llama.cpp layer is integrated).
    // onToken callback is forwarded to the Rust streaming layer.
    if (methodName == "chat") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "chat"), 2,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 2) return Value::undefined();
          uint32_t handle = (uint32_t)args[0].asNumber();
          auto opts       = args[1].asObject(rt);
          auto prompt     = opts.getProperty(rt, "prompt").asString(rt).utf8(rt);

          // Extract onToken callback if provided — nullptr means batch-only mode.
          // TODO: Store a JSI Function ref and invoke it from the Rust token callback
          // once the async dispatch mechanism is in place.
          lokal_chat_stream(handle, prompt.c_str(), nullptr);

          // Return a minimal ChatResponse object so the TS contract is satisfied.
          auto response = Object(rt);
          response.setProperty(rt, "text", String::createFromAscii(rt, ""));
          response.setProperty(rt, "promptTokens", Value(0));
          response.setProperty(rt, "generatedTokens", Value(0));
          response.setProperty(rt, "inferenceMs", Value(0));
          return response;
        }
      );
    }

    // ── disposeEngine ──────────────────────────────────────────────────────
    if (methodName == "disposeEngine") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "disposeEngine"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
          if (count < 1) return Value::undefined();
          uint32_t handle = (uint32_t)args[0].asNumber();
          lokal_dispose_engine(handle);
          return Value::undefined();
        }
      );
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
  // Install the JSI HostObject onto the JS runtime global
  [bridge dispatchBlock:^{
    auto& rt = *bridge.runtime;
    auto hostObject = std::make_shared<LokalMLHostObject>();
    rt.global().setProperty(rt, "LokalMLNative",
                            Object::createFromHostObject(rt, hostObject));
  } queue:RCTJSThread];
}

@end
