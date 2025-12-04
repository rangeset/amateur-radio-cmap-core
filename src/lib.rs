use rayon::prelude::*;
use shapefile::{Shape, ShapeReader};
use std::io::Cursor;
use std::mem::ManuallyDrop;
use wide::CmpLt;
use wide::f64x8;

const DEGREE_TO_RADIAN_CONSTANT: f64 = std::f64::consts::PI / 180_f64;
const RADIAN_TO_DEGREE_CONSTANT: f64 = 180_f64 / std::f64::consts::PI;
const FRAC_PI_DEGREE: f64x8 = f64x8::new([180f64; 8]);

#[unsafe(no_mangle)]
pub extern "C" fn free<T>(ptr: *const T, len: usize) {
    unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr as *mut T, len));
    }
}

#[repr(C)]
pub struct LatlonToAzimnthIsometricCsupport {
    pub x: f64,
    pub y: f64,
}

#[unsafe(no_mangle)]
pub extern "C" fn latlon_to_azimnth_isometric_csupport(
    latitude_delta: f64,
    longitude_delta: f64,
) -> LatlonToAzimnthIsometricCsupport {
    let (x, y): (f64, f64) = latlon_to_azimnth_isometric(latitude_delta, longitude_delta);
    LatlonToAzimnthIsometricCsupport { x, y }
}

pub fn latlon_to_azimnth_isometric(latitude_delta: f64, longitude_delta: f64) -> (f64, f64) {
    let square = |x: f64| x * x;
    let degree_to_radian = |degree: f64| degree * DEGREE_TO_RADIAN_CONSTANT;
    let radian_to_degree = |radian: f64| radian * RADIAN_TO_DEGREE_CONSTANT;
    let latitude_delta_radian: f64 = degree_to_radian(latitude_delta);
    let longitude_delta_radian: f64 = degree_to_radian(longitude_delta);
    let hemispheres_anterior_or_posterior: bool = longitude_delta.abs() < 90_f64;
    let latitude_delta_radian_sine: f64 = latitude_delta_radian.sin();
    let longitude_delta_radian_sine: f64 = longitude_delta_radian.sin();
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
    (
        longitude_delta_radian_sine * k,
        latitude_delta_radian_sine * k,
    )
}

#[repr(C)]
pub struct LatlonToAzimnthIsometricSimdCsupport {
    pub a_ptr: *const f64,
    pub a_len: usize,
    pub b_ptr: *const f64,
    pub b_len: usize,
}

impl LatlonToAzimnthIsometricSimdCsupport {
    pub fn new((a, b): (Vec<f64>, Vec<f64>)) -> Self {
        let mut result: LatlonToAzimnthIsometricSimdCsupport =
            LatlonToAzimnthIsometricSimdCsupport {
                a_ptr: std::ptr::null(),
                a_len: 0usize,
                b_ptr: std::ptr::null(),
                b_len: 0usize,
            };
        result.set_a(a);
        result.set_b(b);
        result
    }

    pub fn get(&self) -> (Vec<f64>, Vec<f64>) {
        (self.get_a(), self.get_b())
    }

    pub fn get_a(&self) -> Vec<f64> {
        unsafe {
            let slice = std::slice::from_raw_parts(self.a_ptr, self.a_len);
            slice.to_vec()
        }
    }

    pub fn set_a(&mut self, data: Vec<f64>) {
        let data_memory = ManuallyDrop::new(data.into_boxed_slice());
        self.a_ptr = data_memory.as_ptr();
        self.a_len = data_memory.len();
    }

    pub fn get_b(&self) -> Vec<f64> {
        unsafe {
            let slice = std::slice::from_raw_parts(self.b_ptr, self.b_len);
            slice.to_vec()
        }
    }

    pub fn set_b(&mut self, data: Vec<f64>) {
        let data_memory = ManuallyDrop::new(data.into_boxed_slice());
        self.b_ptr = data_memory.as_ptr();
        self.b_len = data_memory.len();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn latlon_to_azimnth_isometric_simd_csupport(
    parameter: LatlonToAzimnthIsometricSimdCsupport,
) -> LatlonToAzimnthIsometricSimdCsupport {
    LatlonToAzimnthIsometricSimdCsupport::new(latlon_to_azimnth_isometric_simd(
        parameter.get_a(),
        parameter.get_b(),
    ))
}

pub fn latlon_to_azimnth_isometric_simd(
    latitude_delta_vec: Vec<f64>,
    longitude_delta_vec: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let len = std::cmp::min(latitude_delta_vec.len(), longitude_delta_vec.len());
    let latitude_delta_vec_chunks = latitude_delta_vec[..len].par_chunks_exact(8);
    let longitude_delta_vec_chunks = longitude_delta_vec[..len].par_chunks_exact(8);
    let chunks_remainder = latitude_delta_vec_chunks
        .remainder()
        .iter()
        .copied()
        .zip(longitude_delta_vec_chunks.remainder().iter().copied());
    let chunks_bodys: Vec<([f64; 8], [f64; 8])> = latitude_delta_vec_chunks
        .zip(longitude_delta_vec_chunks)
        .map(
            |(latitude_slice, longitude_slice): (&[f64], &[f64])| -> ([f64; 8], [f64; 8]) {
                let latitude_array = latitude_slice;
                let longitude_array = longitude_slice;

                let square = |x: f64x8| x * x;

                let latitude = f64x8::from(latitude_array);
                let longitude = f64x8::from(longitude_array);

                let posistion = longitude.abs().simd_lt(f64x8::FRAC_PI_2);

                let latitude_radian = latitude.to_radians();
                let longitude_radian = longitude.to_radians();

                let latitude_radian_sine = latitude_radian.sin();
                let longitude_radian_sine = longitude_radian.sin();

                let distance =
                    (square(latitude_radian_sine) + square(longitude_radian_sine)).sqrt();

                let distance_arcsine = distance.asin().to_degrees();
                let distance_arcsine_back = FRAC_PI_DEGREE - distance_arcsine;

                let k_uncheck = posistion.blend(distance_arcsine, distance_arcsine_back) / distance;

                let k_infinite_mask = k_uncheck.is_inf();
                let k_nan_mask = k_uncheck.is_nan();

                let k_bad = k_infinite_mask | k_nan_mask;

                let k = k_bad.blend(f64x8::ZERO, k_uncheck);

                ((latitude * k).to_array(), (longitude * k).to_array())
            },
        )
        .collect();

    let mut xs: Vec<f64> = Vec::with_capacity(len);
    let mut ys: Vec<f64> = Vec::with_capacity(len);

    for (x, y) in chunks_bodys {
        xs.extend(x);
        ys.extend(y);
    }

    for (latitude, longitude) in chunks_remainder {
        let result = latlon_to_azimnth_isometric(latitude, longitude);
        xs.push(result.0);
        ys.push(result.1);
    }

    (xs, ys)
}

#[repr(C)]
pub struct ReturnContent {
    pub status: bool,
    pub ptr: *const u8,
    pub len: usize,
}

impl ReturnContent {
    fn new(data: Vec<u8>, status: bool) -> Self {
        let data = ManuallyDrop::new(data.into_boxed_slice());
        ReturnContent {
            status: status,
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }

    fn new_result(result: Result<Vec<u8>, shapefile::Error>) -> Self {
        match result {
            Ok(data) => Self::new(data, true),
            Err(e) => Self::new(e.to_string().into_bytes(), false),
        }
    }
}

#[repr(C)]
pub struct GenerateParameters {
    pub color_point: u8,
    pub color_multipoint: u8,
    pub color_line: u8,
    pub color_polygon: u8,
    pub width_point: u8,
    pub width_multipoint: u8,
    pub width_line: u8,
    pub width_polygon: u8,
    pub fineness: u8,
}

#[unsafe(no_mangle)]
pub fn shapefile_generate_csupport(
    buffer_ptr: *const u8,
    buffer_len: usize,
    parameter: GenerateParameters,
) -> ReturnContent {
    ReturnContent::new_result(shapefile_generate(
        unsafe { std::slice::from_raw_parts(buffer_ptr, buffer_len) },
        parameter,
    ))
}

pub fn shapefile_generate(
    buffer: &[u8],
    parameter: GenerateParameters,
) -> Result<Vec<u8>, shapefile::Error> {
    let cursor = Cursor::new(buffer);
    let mut reader = match ShapeReader::new(cursor) {
        Ok(data) => data,
        Err(e) => return Err(e),
    };
    // TODO Draw the picture used par_iter
    //reader.for_each(|shape| )
    ReturnContent::new(Vec::new(), true)
}

// TODO Draw fuction
//fn shapefile_draw() -> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;
    use approx::assert_abs_diff_eq;
    use std::time::Instant;

    struct ResultBody<T: AbsDiffEq> {
        totals: Vec<T>,
        results: Vec<T>,
        epsilon: f64,
    }

    impl<T: AbsDiffEq<Epsilon = f64> + std::fmt::Debug> ResultBody<T> {
        fn new(epsilon: f64) -> Self {
            ResultBody {
                totals: Vec::new(),
                results: Vec::new(),
                epsilon,
            }
        }
        fn add_tuple_2(&mut self, (total_a, total_b): (T, T), (result_a, result_b): (T, T)) {
            self.totals.push(total_a);
            self.totals.push(total_b);
            self.results.push(result_a);
            self.results.push(result_b);
        }
        fn compare(&self) {
            assert_abs_diff_eq!(
                self.totals.as_slice(),
                self.results.as_slice(),
                epsilon = self.epsilon
            );
            println!("Compared");
        }
    }

    #[test]
    fn transprojection() {
        let mut result = ResultBody::<f64>::new(1e-12_f64);
        let start = Instant::now();
        let standard = start.elapsed();

        result.add_tuple_2(
            (0000_f64, 0000_f64),
            latlon_to_azimnth_isometric(000_f64, 0000_f64),
        );
        result.add_tuple_2(
            (0000_f64, -090_f64),
            latlon_to_azimnth_isometric(-90_f64, 0000_f64),
        );
        result.add_tuple_2(
            (0000_f64, 0135_f64),
            latlon_to_azimnth_isometric(045_f64, 0180_f64),
        );

        let end_ltai = start.elapsed();

        // TODO add test for ltais
        //latlon_to_azimnth_isometric_simd()

        result.compare();

        println!("standard: {:?}\nltai: {:?}", standard, end_ltai);
    }
}
