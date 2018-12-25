//! Backend for vector operations

// trait

// [[file:~/Workspace/Programming/rust-libs/vecfx/math.org::*trait][trait:1]]
#[cfg(feature = "nalgebra")]
use nalgebra::DVector;

/// Abstracting simple vector based math operations
pub trait FloatVectorMath {
    /// y += c*x
    fn vecadd(&mut self, x: &[f64], c: f64);

    /// vector dot product
    /// s = x.dot(y)
    fn vecdot(&self, other: &[f64]) -> f64;

    /// y = z
    fn veccpy(&mut self, x: &[f64]);

    /// y = -x
    fn vecncpy(&mut self, x: &[f64]);

    /// z = x - y
    fn vecdiff(&mut self, x: &[f64], y: &[f64]);

    /// y *= c
    fn vecscale(&mut self, c: f64);

    /// ||x||
    fn vec2norm(&self) -> f64;

    /// 1 / ||x||
    fn vec2norminv(&self) -> f64;

    /// Minimum value of the samples.
    fn min(&self) -> f64;

    /// Maximum value of the samples.
    fn max(&self) -> f64;

    #[cfg(feature = "nalgebra")]
    /// Create dynamically allocated column vector from self
    fn to_column_vector(&self) -> DVector<f64>;

    #[cfg(feature = "nalgebra")]
    fn to_dvector(&self) -> DVector<f64> {
        self.to_column_vector()
    }
}

impl FloatVectorMath for [f64] {
    /// y += c*x
    fn vecadd(&mut self, x: &[f64], c: f64) {
        for (y, x) in self.iter_mut().zip(x) {
            *y += c * x;
        }
    }

    /// s = y.dot(x)
    fn vecdot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).map(|(x, y)| x * y).sum()
    }

    /// y *= c
    fn vecscale(&mut self, c: f64) {
        for y in self.iter_mut() {
            *y *= c;
        }
    }

    /// y = x
    fn veccpy(&mut self, x: &[f64]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = *x;
        }
    }

    /// y = -x
    fn vecncpy(&mut self, x: &[f64]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = -x;
        }
    }

    /// z = x - y
    fn vecdiff(&mut self, x: &[f64], y: &[f64]) {
        for ((z, x), y) in self.iter_mut().zip(x).zip(y) {
            *z = x - y;
        }
    }

    /// ||x||
    fn vec2norm(&self) -> f64 {
        let n2 = self.vecdot(&self);
        n2.sqrt()
    }

    /// 1/||x||
    fn vec2norminv(&self) -> f64 {
        1.0 / self.vec2norm()
    }

    fn min(&self) -> f64 {
        assert!(!self.is_empty());
        self.iter().fold(self[0], |p, q| p.min(*q))
    }

    fn max(&self) -> f64 {
        assert!(!self.is_empty());
        self.iter().fold(self[0], |p, q| p.max(*q))
    }

    #[cfg(feature = "nalgebra")]
    fn to_column_vector(&self) -> DVector<f64> {
        DVector::from_column_slice(self.len(), &self)
    }
}

#[test]
fn test_vec_math() {
    // vector scaled add
    let x = [1.0, 1.0, 1.0];
    let c = 2.;

    let mut y = [1.0, 2.0, 3.0];
    y.vecadd(&x, c);

    assert_eq!(3.0, y[0]);
    assert_eq!(4.0, y[1]);
    assert_eq!(5.0, y[2]);

    // vector dot
    let v = y.vecdot(&x);
    assert_eq!(12.0, v);

    // vector scale
    y.vecscale(2.0);
    assert_eq!(6.0, y[0]);
    assert_eq!(8.0, y[1]);
    assert_eq!(10.0, y[2]);

    // vector diff
    let mut z = y.clone();
    z.vecdiff(&x, &y);
    assert_eq!(-5.0, z[0]);
    assert_eq!(-7.0, z[1]);
    assert_eq!(-9.0, z[2]);

    // vector copy
    y.veccpy(&x);

    // y = -x
    y.vecncpy(&x);
    assert_eq!(-1.0, y[0]);
    assert_eq!(-1.0, y[1]);
    assert_eq!(-1.0, y[2]);
}

#[cfg(feature = "nalgebra")]
#[test]
fn test_vec_math_na() {
    // nalgebra
    let y = [-1.0; 3];
    let v = y.to_column_vector();
    assert_eq!(v.norm_squared(), 3.0);
}
// trait:1 ends here
