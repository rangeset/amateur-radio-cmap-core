use shapefile::{Shape, ShapeReader};
use std::io::Cursor;
use std::mem::ManuallyDrop;

const DEGREE_TO_RADIAN_CONSTANT: f64 = std::f64::consts::PI / 180_f64;
const RADIAN_TO_DEGREE_CONSTANT: f64 = 180_f64 / std::f64::consts::PI;

pub fn latlon_to_amateur_radio_great_circle_map(
    latitude_delta: f64,
    longitude_delta: f64,
) -> (f64, f64) {
    let square = |x: f64| x * x;
    let degree_to_radian = |degree: f64| degree * DEGREE_TO_RADIAN_CONSTANT;
    let radian_to_degree = |radian: f64| radian * RADIAN_TO_DEGREE_CONSTANT;
    let latitude_delta_radian: f64 = degree_to_radian(latitude_delta);
    let longitude_delta_radian: f64 = degree_to_radian(longitude_delta);
    let hemispheres_anterior_or_posterior: bool = longitude_delta.abs() < 90_f64;

    #[cfg(test)]
    dbg!(latitude_delta_radian, longitude_delta_radian);

    let latitude_delta_radian_sine: f64 = latitude_delta_radian.sin();
    let longitude_delta_radian_sine: f64 = longitude_delta_radian.sin();

    #[cfg(test)]
    dbg!(latitude_delta_radian_sine, longitude_delta_radian_sine);

    let distance: f64 =
        (square(latitude_delta_radian_sine) + square(longitude_delta_radian_sine)).sqrt();
    let k: f64 = match distance {
        0_f64 => 0_f64,
        _ => {
            (if hemispheres_anterior_or_posterior {
                radian_to_degree(distance.asin())
            } else {
                180_f64 - radian_to_degree(distance.asin())
            }) / distance
        }
    };

    #[cfg(test)]
    dbg!(distance, k);

    (
        longitude_delta_radian_sine * k,
        latitude_delta_radian_sine * k,
    )
}

// TODO used wide to accelerate multipoint,line,polygon
//pub fn latlon_to_amateur_radio_great_circle_map_simd() {}

pub struct ReturnContent {
    pub status: bool,
    pub ptr: *const u8,
    pub len: usize,
}

impl ReturnContent {
    fn new(data: Vec<u8>, status: bool) -> Self {
        let data = ManuallyDrop::new(data);
        ReturnContent {
            status: status,
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }
}

pub struct ColorData {
    pub color_point: u8,
    pub color_multipoint: u8,
    pub color_line: u8,
    pub color_polygon_line: u8,
    pub color_polygon_fill: u8,
}

pub fn shapefile_generate(
    buffer_ptr: *const u8,
    buffer_len: usize,
    color: ColorData,
) -> ReturnContent {
    let buffer = unsafe { std::slice::from_raw_parts(buffer_ptr, buffer_len) };
    let cursor = Cursor::new(buffer);
    let mut reader = match ShapeReader::new(cursor) {
        Ok(data) => data,
        Err(e) => return ReturnContent::new(e.to_string().into_bytes(), false),
    };
    // TODO Draw the picture used par_iter
    //    reader.for_each(|shape| )
    ReturnContent::new(Vec::new(), true)
}

// TODO Draw fuction
//fn shapefile_draw() -> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transprojection() {
        let result = latlon_to_amateur_radio_great_circle_map(0_f64, 0_f64);
        assert_eq!(result, (0_f64, 0_f64));

        let result = latlon_to_amateur_radio_great_circle_map(-90_f64, 0_f64);
        assert_eq!(result, (0_f64, -90_f64));

        let result = latlon_to_amateur_radio_great_circle_map(0_f64, 180_f64);
        assert_eq!(result, (180_f64, 0_f64));
    }
}
