from PIL import Image, ImageOps

from training.prepare_native_scale_heads import pixels


def test_duplicate_audit_detects_original_exif_photo_and_upright_copy(tmp_path):
    image = Image.new("RGB", (3, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (50, 10, 20), (100, 20, 40), (150, 30, 60)])
    exif = image.getexif()
    exif[274] = 6
    original = tmp_path / "original.png"
    image.save(original, exif=exif)
    upright = tmp_path / "upright.png"
    with Image.open(original) as decoded:
        ImageOps.exif_transpose(decoded).save(upright)
    assert Image.open(original).size != Image.open(upright).size
    assert pixels(original) == pixels(upright)
