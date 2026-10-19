// Standalone stand-in for Skia's SkBitmap, so ml::InputPiece is a complete
// type. The backend ignores image pieces. The Chromium build uses real Skia.
#ifndef CHROMEML_STANDALONE_SKBITMAP_H_
#define CHROMEML_STANDALONE_SKBITMAP_H_

class SkBitmap {
 public:
  void* unused[8] = {};
};

#endif  // CHROMEML_STANDALONE_SKBITMAP_H_
