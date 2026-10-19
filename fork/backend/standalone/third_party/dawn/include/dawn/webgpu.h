// Standalone stand-in for Dawn's generated webgpu.h. It declares only the
// types chrome_ml_api.h names. The enums keep Dawn's 32-bit width so struct
// layouts match. The Chromium build uses the real header instead.
#ifndef CHROMEML_STANDALONE_WEBGPU_H_
#define CHROMEML_STANDALONE_WEBGPU_H_

typedef struct WGPUAdapterImpl* WGPUAdapter;
typedef enum WGPUAdapterType { WGPUAdapterType_Force32 = 0x7FFFFFFF } WGPUAdapterType;
typedef enum WGPUBackendType { WGPUBackendType_Force32 = 0x7FFFFFFF } WGPUBackendType;

#endif  // CHROMEML_STANDALONE_WEBGPU_H_
