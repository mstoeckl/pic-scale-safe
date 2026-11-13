use image::{
    DynamicImage::{self, ImageRgba8},
    GenericImage, Pixel, RgbaImage,
};
use pic_scale_safe::*;

/// ```txt
/// ****. . .
/// **** . .
/// ****. . .
/// @
/// @@
/// @@@
/// ```
fn render_test<const CHANNELS: usize>() {
    // TODO: render to format X
    //
}

enum Crop {
    None,
    Horiz,
    Vert,
    Center,
}

#[test]
fn snapshot() {
    for (crop_name, crop) in [
        ('n', Crop::None),
        ('h', Crop::Horiz),
        ('v', Crop::Vert),
        ('c', Crop::Center),
    ] {
        for size in [(100, 100), (150, 50), (50, 150)] {
            for scale in [
                0.1,
                0.33333333333333333333333,
                0.5,
                0.77,
                1.0f64,
                1.234567890123456789,
                2.0,
                4.444444444444444444,
            ] {
                for (resamp_name, resample) in [
                    ("bilinear", ResamplingFunction::Bilinear),
                    ("nearest", ResamplingFunction::Nearest),
                    ("area", ResamplingFunction::Area),
                    ("gaussian", ResamplingFunction::Gaussian),
                ] {
                    let out_size = (
                        (size.0 as f64 * scale).round() as u32,
                        (size.1 as f64 * scale).round() as u32,
                    );

                    let mut image =
                        image::DynamicImage::new(size.0, size.1, image::ColorType::Rgba8);

                    for x in 0..size.0 / 2 {
                        for y in 0..size.1 / 2 {
                            image.put_pixel(x, y, *Pixel::from_slice(&[255, 0, 0, 100]));
                        }
                    }
                    for x in size.0 / 2..size.0 {
                        for y in 0..size.1 / 2 {
                            image.put_pixel(
                                x,
                                y,
                                *if x.wrapping_add(y) % 2 == 0 {
                                    Pixel::from_slice(&[0, 0, 0, 200])
                                } else {
                                    Pixel::from_slice(&[255, 255, 255, 255])
                                },
                            );
                        }
                    }
                    for x in 0..size.0 {
                        for y in size.1 / 2..size.1 {
                            image.put_pixel(
                                x,
                                y,
                                *if x > y - size.1 / 2 {
                                    Pixel::from_slice(&[255, 255, 255, 255])
                                } else {
                                    Pixel::from_slice(&[0, 120, 120, 255])
                                },
                            );
                        }
                    }

                    let image = image.to_rgba8();

                    let fpmult = |v: u32, s: u8| -> FractIndex {
                        let w = ((u128::from(v) << 32) * u128::from(s) + 5) / 10;
                        FractIndex {
                            int: (w >> 32) as usize,
                            frac: (w & ((1 << 32) - 1)) as u32,
                        }
                    };
                    let cmult = |v, s| (v * s + 5) / 10;

                    // TODO: exactly preserve aspect ratio, instead, so pick out_size first
                    let (source, out_size) = match crop {
                        Crop::None => (
                            CropRect::new_integral(0, size.0 as usize, 0, size.1 as usize),
                            out_size,
                        ),
                        Crop::Horiz => (
                            CropRect {
                                x1: fpmult(size.0, 0),
                                x2: fpmult(size.0, 10),
                                y1: fpmult(size.1, 3),
                                y2: fpmult(size.1, 6),
                            },
                            (out_size.0, cmult(out_size.1, 3)),
                        ),
                        Crop::Vert => (
                            CropRect {
                                x1: fpmult(size.0, 4),
                                x2: fpmult(size.0, 7),
                                y1: fpmult(size.1, 0),
                                y2: fpmult(size.1, 10),
                            },
                            (cmult(out_size.0, 3), out_size.0),
                        ),
                        Crop::Center => (
                            CropRect {
                                x1: fpmult(size.0, 1),
                                x2: fpmult(size.0, 9),
                                y1: fpmult(size.1, 3),
                                y2: fpmult(size.1, 8),
                            },
                            (cmult(out_size.0, 8), cmult(out_size.0, 5)),
                        ),
                    };

                    let name = format!(
                        "snapshots/{}_{}x{}_{}x{}_{}.png",
                        crop_name, size.0, size.1, out_size.0, out_size.1, resamp_name
                    );
                    println!("Creating {}", name);

                    let out = crop_resize_rgba8(
                        image.as_raw().as_slice(),
                        ImageSize::new(size.0 as usize, size.1 as usize),
                        source,
                        ImageSize::new(out_size.0 as usize, out_size.1 as usize),
                        resample,
                    )
                    .unwrap();

                    let resized = RgbaImage::from_raw(out_size.0, out_size.1, out)
                        .expect("resize output should match");

                    resized.save(name).unwrap();
                }
            }
        }
    }
}
