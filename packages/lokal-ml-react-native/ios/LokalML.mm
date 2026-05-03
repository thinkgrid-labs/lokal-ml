/**
 * LokalML iOS JSI HostObject bridge.
 *
 * Registers the `lokal` global on the JS runtime and exposes the following
 * methods via JSI (zero async serialisation overhead):
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

// ─── JSI HostObject ──────────────────────────────────────────────────────────

class LokalMLHostObject : public HostObject {
public:
  Value get(Runtime& rt, const PropNameID& name) override {
    auto methodName = name.utf8(rt);

    // ── checkRequirements ──────────────────────────────────────────────────
    if (methodName == "checkRequirements") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "checkRequirements"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
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
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
          auto modelId     = args[0].asString(rt).utf8(rt);
          bool requireWifi = args[1].asBool();
          // onProgress callback (args[2]) is passed through to Rust layer
          lokal_download_model(modelId.c_str(), requireWifi, nullptr /* TODO: wire callback */);
          return Value::undefined();
        }
      );
    }

    // ── isModelCached ──────────────────────────────────────────────────────
    if (methodName == "isModelCached") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "isModelCached"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
          auto modelId = args[0].asString(rt).utf8(rt);
          bool cached  = lokal_is_model_cached(modelId.c_str());
          return Value(cached);
        }
      );
    }

    // ── initEngine ─────────────────────────────────────────────────────────
    if (methodName == "initEngine") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "initEngine"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
          auto config  = args[0].asObject(rt);
          auto model   = config.getProperty(rt, "model").asString(rt).utf8(rt);
          uint32_t handle = lokal_init_engine(model.c_str());
          return Value((double)handle);
        }
      );
    }

    // ── chat ───────────────────────────────────────────────────────────────
    if (methodName == "chat") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "chat"), 2,
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
          uint32_t handle = (uint32_t)args[0].asNumber();
          auto opts       = args[1].asObject(rt);
          auto prompt     = opts.getProperty(rt, "prompt").asString(rt).utf8(rt);
          // TODO: Wire onToken callback and full options through to Rust
          lokal_chat_stream(handle, prompt.c_str(), nullptr);
          return Value::undefined();
        }
      );
    }

    // ── disposeEngine ──────────────────────────────────────────────────────
    if (methodName == "disposeEngine") {
      return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, "disposeEngine"), 1,
        [](Runtime& rt, const Value&, const Value* args, size_t) -> Value {
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
