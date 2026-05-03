Pod::Spec.new do |s|
  s.name             = 'lokal-ml-react-native'
  s.version          = '0.1.0'
  s.summary          = 'On-device SLM inference for React Native via JSI'
  s.homepage         = 'https://lokal-ml.dev'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.authors          = { 'thinkgrid-labs' => 'hello@thinkgrid.dev' }
  s.source           = { :git => 'https://github.com/thinkgrid-labs/lokal-ml.git', :tag => s.version }
  s.platforms        = { :ios => '16.0' }

  s.source_files     = 'cpp/**/*.{h,cpp}', 'ios/**/*.{h,m,mm}'

  # Pre-built XCFramework (Rust static library for arm64 + sim)
  # Built by scripts/build-ios.sh — attached to each GitHub Release
  s.vendored_frameworks = 'ios/LokalMLRust.xcframework'

  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'OTHER_LDFLAGS' => '-lc++',
  }

  s.dependency 'React-Core'
  s.dependency 'React-jsi'
end
