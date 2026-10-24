# Provenance - non-code assets

Record of where this repository's image assets came from, per CireSnave's provenance rule
(`C:\Projects\CIRESNAVE-EXPECTATIONS.md` section 6.4a, 2026-09-17). Measured at `origin/main` `9a5608c6`, 2026-10-24.
Documentation only: no asset was changed.

| file | maker / tool | date | licence |
|---|---|---|---|
| `candlelight.gif` (sha256 `5e909b7db0e102f8f8b81f6f9fdc702e8f8738eacde346c1b181ac37f8fd91c0`, 678297 bytes) | Generated for this project by the same AI tool as vulkane's logo. CireSnave (2026-10-24, verbatim): "Candlelight's Candle was custom generated for us by the same LLM as Vulkane.  I don't know more than that about those." The file carries no embedded provenance marker (see below). | First committed 2026-04-02 (`d70a03f325294a7c3ac354dc74d4768c4c96db5e`, "v0.2.2: Fix cudnn module, update to Candle v0.10.2, add ML utility crates") | no rights claimed by the generating tool per CireSnave; the output terms of the service were not checked here (pending CireSnave, decision 267) |

## Embedded-marker check

Checked: a byte search of the whole GIF for `c2pa`, `jumb`, `XMP`, `xpacket`, `Exif`, `Microsoft`, `Designer`,
`trainedAlgorithmicMedia`, `NETSCAPE`, `ImageMagick`: no hits (GIF89a header; no APP11/JUMBF-style container exists in GIF).
**Control:** the same style of check on vulkane's `vulkan.jpeg` finds `jumb` at offset 18, `c2pa` at 30 and `c2cl` at 1207, and
a segment parse of it yields a C2PA manifest naming `Image Creator from Designer` (recorded in vulkane's `PROVENANCE.md`).
The GIF therefore has no machine-readable evidence of its tool, and "same tool as vulkane's logo" rests on CireSnave's
statement alone. Absence of a marker does not show the file was not AI-generated, or that it was; a GIF may simply lose such metadata.

**Method:** first-add commit found with `git log --follow --diff-filter=A -- candlelight.gif` (one result, above); sha256 taken
from `git show origin/main:candlelight.gif`; `origin/main` read at `9a5608c6a63ebf5e4094e2c50610ba496c244d6f`. The repository declares `MIT OR Apache-2.0`
(`Cargo.toml`; `LICENSE-MIT`, `LICENSE-APACHE`); that declaration does not itself license this image.
