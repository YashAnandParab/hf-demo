/** Self-check for the SSE reader. The tricky part is not parsing one frame,
 *  it is not losing a token when the network splits a frame in half.
 *
 *      node --experimental-strip-types src/lib/sse.check.ts
 */
import assert from "node:assert/strict";
import { decodeFrame, framePayload, splitFrames } from "./sse.ts";

// A frame that arrives whole.
{
  const { frames, rest } = splitFrames('data: {"token":"hi"}\n\n');
  assert.equal(frames.length, 1);
  assert.equal(rest, "");
  assert.deepEqual(decodeFrame(framePayload(frames[0])), [
    { type: "token", text: "hi" },
  ]);
}

// A frame split across two reads must survive, and only once.
{
  let buf = "";
  const out: string[] = [];
  for (const chunk of ['data: {"tok', 'en":"he"}\n\ndata: {"token":"llo"}\n\n']) {
    buf += chunk;
    const { frames, rest } = splitFrames(buf);
    buf = rest;
    for (const f of frames)
      for (const ev of decodeFrame(framePayload(f)))
        if (ev.type === "token") out.push(ev.text);
  }
  assert.equal(out.join(""), "hello");
  assert.equal(buf, "", "no partial frame should be left over");
}

// CRLF framing, multi-line data, and the terminator.
assert.equal(splitFrames("data: a\r\n\r\ndata: b").frames.length, 1);
assert.equal(framePayload("event: x\ndata: one\ndata: two"), "one\ntwo");
assert.deepEqual(decodeFrame("[DONE]"), [{ type: "done" }]);

// Alternate token keys, sources, and junk that must not kill the stream.
assert.deepEqual(decodeFrame('{"delta":"x"}'), [{ type: "token", text: "x" }]);
assert.deepEqual(decodeFrame('{"sources":[{"id":"1"}]}'), [
  { type: "sources", sources: [{ id: "1" }] as never },
]);
assert.deepEqual(decodeFrame(""), []);
assert.deepEqual(decodeFrame('{"token":""}'), []);
assert.deepEqual(decodeFrame("plain text"), [
  { type: "token", text: "plain text" },
]);

console.log("sse: ok");
