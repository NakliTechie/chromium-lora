// Standalone stand-in for Dawn's proc table. The backend never reads it.
#ifndef CHROMEML_STANDALONE_DAWN_PROC_TABLE_H_
#define CHROMEML_STANDALONE_DAWN_PROC_TABLE_H_

typedef struct DawnProcTable {
  void* unused;
} DawnProcTable;

#endif  // CHROMEML_STANDALONE_DAWN_PROC_TABLE_H_
