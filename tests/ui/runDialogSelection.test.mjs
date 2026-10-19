import assert from "node:assert/strict";
import { test } from "node:test";

import { defaultCapabilityIds, isSelectedByDefault, rememberedCapabilityIds } from "../../extensions/AI.Core/ui/runDialogSelection.js";

const tagging = { capabilityId: "tagging" };
const faces = { capabilityId: "faces", selectedByDefault: true };
const shots = { capabilityId: "shots", selectedByDefault: false };

test("features are selected by default unless they say otherwise", () => {
  assert.equal(isSelectedByDefault(tagging), true);
  assert.equal(isSelectedByDefault(faces), true);
  assert.equal(isSelectedByDefault(shots), false);
  assert.deepEqual(defaultCapabilityIds([tagging, shots, faces]), ["tagging", "faces"]);
  assert.deepEqual(defaultCapabilityIds(undefined), []);
});

test("a remembered selection never brings back a feature that is not selected by default", () => {
  assert.deepEqual(rememberedCapabilityIds(["shots", "tagging"], [tagging, shots, faces]), ["tagging"]);
});

test("a remembered selection keeps its order and drops features no longer offered", () => {
  assert.deepEqual(rememberedCapabilityIds(["faces", "visual", "tagging", 7], [tagging, faces]), ["faces", "tagging"]);
  assert.deepEqual(rememberedCapabilityIds(null, [tagging]), []);
});

test("a newly installed feature that is not selected by default stays unticked", () => {
  // Everything that was offered when the selection was remembered, and a new opt-in feature since.
  const remembered = ["tagging", "faces"];
  assert.deepEqual(rememberedCapabilityIds(remembered, [tagging, faces, shots]), ["tagging", "faces"]);
  assert.deepEqual(defaultCapabilityIds([tagging, faces, shots]), ["tagging", "faces"]);
});
