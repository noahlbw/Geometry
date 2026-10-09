# View wording clarification

The initially deployed protocol.md mislabeled overlap128 as stride128.
The unchanged shared evaluator uses tile_starts(length,512,128), where128 is
overlap and nominal step is384, with end-aligned border windows.
This is a documentation correction, not a model/protocol change. Keep the
original remote protocol.md and JSON frozen; retain this clarification alongside
them. No evaluation was restarted and no inference setting was altered.
