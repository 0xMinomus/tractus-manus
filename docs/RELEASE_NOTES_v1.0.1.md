# Tractus Manus 1.0.1

This is a documentation and visual refresh. Runtime behavior is unchanged.

## Download

Download `tractus-manus-1.0.1-windows.zip`, extract it, and run `setup.bat`
once. Then run `run.bat`. The release includes the MediaPipe hand landmark
model.

## Visual refresh

- Monochrome halftone hand artwork made from hand landmark points.
- Bodoni Moda for display type and IBM Plex Mono for technical labels.
- A black, white, and gray README header and system map.
- Font source files and SIL Open Font License texts in the archive.
- No camera images or face data are included.

## Runtime

The gesture engine, threshold defaults, optional classifier format, and Windows
input code are unchanged from 1.0.0.

## Privacy

The runtime processes frames in memory. The optional collector stores only hand
landmark coordinates and labels. This release does not include training
recordings, face images, or a personal classifier model.

## Checksum

Verify the downloaded ZIP against the `.sha256` file attached to the release.
