const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export async function pngHasAlphaChannel(file) {
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.length < 33) return false;
  if (!PNG_SIGNATURE.every((v, i) => bytes[i] === v)) return false;
  const view = new DataView(arrayBuffer);
  const ihdrLength = view.getUint32(8);
  const ihdrType = String.fromCharCode(...bytes.slice(12, 16));
  if (ihdrLength !== 13 || ihdrType !== "IHDR") return false;
  const colorType = bytes[25];
  if (colorType === 4 || colorType === 6) return true;
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const chunkLength = view.getUint32(offset);
    const chunkType = String.fromCharCode(...bytes.slice(offset + 4, offset + 8));
    if (chunkType === "tRNS") return true;
    offset += chunkLength + 12;
  }
  return false;
}
