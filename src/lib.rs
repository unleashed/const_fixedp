// Copyright (c) 2021 - Alejandro Martinez Ruiz <alex@flawedcode.org>
//
// const_fixedp - A crate to handle const unsigned fixed point values.
//
// So far this only includes addition and subtraction, with no (direct)
// conversions from f32/f64, and the internal type used is set to `u64`.
//
// There is optional support for `serde` serialization and deserialization.
//
// Nothing of the above _has to_ be that way, so contributions are welcome!
//
#![deny(clippy::all, clippy::cargo)]
#![doc(html_playground_url = "https://play.rust-lang.org/")]
// Force all constant evaluations to be successful or fail compilation.
#![forbid(const_err)]

//! Basic unsigned fixed-point arithmetic.
//!
//! This module provides a `Copy` type that holds arbitrary precision for
//! fractional values, performs additions and subtractions, and can be
//! compared and sorted efficiently.
//!
//! The internal representation uses 64 bits, and will adapt to the
//! requested precision by reducing the integer range, so keep this in
//! mind when increasing precision but still requiring big values, as
//! compilation will fail if there's not enough bit space for at least
//! [`FixedP::<P>::MIN_UNIT_SCALE`] integer values.
//!
//! Enable the `serde` feature flag to have serialization and
//! deserialization capabilities for this type.
//!

// Macros to force a compilation failure in const evaluation contexts.
#[macro_use]
mod const_panic;

// Serialization and deserialization implementations.
#[cfg(feature = "serde")]
mod serde;

use core::convert::{Infallible, TryFrom};
use core::num::{ParseIntError, TryFromIntError};

/// Type describing the precision required for the fractional part of a
/// fixed-point value.
pub type Precision = u8;

/// An error type covering the cases where units or fractional values are
/// out of range, when a conversion error happens (ie. trying to convert
/// from a negative or too big value), or when parsing fails.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error<const P: Precision> {
    /// Failed to convert an integer type into a `FixedP`.
    #[error("failed to convert from integer: {0}")]
    ConversionError(#[from] TryFromIntError),
    /// Units are too big.
    #[error("units out of range (max {} bits)", FixedP::<P>::UNIT_BITS)]
    OutOfRangeUnits,
    /// Fractional part is too big, ie. over the given scale.
    #[error("fraction out of range (max {} bits, {} base-10 digits)", FixedP::<P>::FRAC_BITS, P)]
    OutOfRangeFrac,
    /// Error parsing a fixed point value. This usually boils down to very
    /// big integers or fractions, or non-digit characters in the string
    /// other than the single `.` separator occurrence between integers
    /// and fractions.
    #[error("failed to parse fixed point value: {0}")]
    ParserError(#[from] ParseIntError),
}

// Allow the above Error to be converted from an infallible type.
// This lets the compiler unify types when a fallible conversion
// turns out to be infallible for a specific source type.
//
// See https://github.com/dtolnay/thiserror/issues/62.
impl<const P: Precision> From<Infallible> for Error<P> {
    fn from(i: Infallible) -> Self {
        match i {}
    }
}

/// A `Copy` type representing an unsigned, fixed point value, supporting
/// const context evaluation.
///
/// Equality and comparisons are fast operations.
///
/// **Note**: This type currently only supports addition and subtraction,
///           and all operations must happen between equivalent types,
///           ie. can't mix `FixedP<5>` with `FixedP<4>`, although in some
///           cases you can convert from/to different precision types.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), const_fixedp::Error<2>> {
/// # use const_fixedp::FixedP;
/// /// Create 11.50 and 1.05 as fixed point values
/// let fixed_point_11_5 = FixedP::<2>::from_units_frac(11, 50)?;
/// let fixed_point_1_05 = FixedP::<2>::from_units_frac(1, 5)?;
///
/// /// Assert that the addition equals 12.55
/// let addition = fixed_point_11_5 + fixed_point_1_05;
/// assert_eq!(addition.units(), 12);
/// assert_eq!(addition.frac(), 55);
/// assert_eq!(&format!("{}", addition), "12.55");
/// # Ok(()) }
/// ```
///
/// The example below fails because we are specifying a fractional part
/// that cannot be represented with the given precision:
///
/// ```should_panic
/// # fn main() -> Result<(), const_fixedp::Error<2>> {
/// # use const_fixedp::FixedP;
/// /// The fractional part cannot represent 3 digits.
/// let fixed_point_1_123 = FixedP::<2>::from_units_frac(1, 123)?;
/// # Ok(()) }
/// ```
///
/// The following example should fail to compile because the requirements
/// on the precision of the fractional part are so big that the units
/// can no longer represent [`FixedP::<P>::MIN_UNIT_SCALE`]:
///
/// ```compile_fail
/// # fn main() -> Result<(), const_fixedp::Error<16>> {
/// # use const_fixedp::FixedP;
/// /// Try to create a fixed-point value with too much precision
/// /// so that the remaining integer part becomes too small.
/// let fixed_point_1_123 = FixedP::<16>::from_units_frac(1, 1)?;
/// # Ok(()) }
/// ```
///
/// Conversely, trying to create a fixed point value with a precision of
/// zero digits also fails to compile:
///
/// ```compile_fail
/// # fn main() -> Result<(), const_fixedp::Error<0>> {
/// # use const_fixedp::FixedP;
/// let fixed_point_1_123 = FixedP::<0>::from_units(1)?;
/// # Ok(()) }
/// ```
///
/// If you ever have two fixed point values with different precision
/// there are still options for you to operate with them as long as
/// you can convert one to the other:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use const_fixedp::FixedP;
/// let fp500_15_p2 = FixedP::<2>::from_units_frac(500, 15)?;
/// let fp1_00001_p5 = FixedP::<5>::from_units_frac(1, 1)?;
///
/// let fp500_15_p5 = FixedP::<5>::try_from_fixed(fp500_15_p2)?;
/// let addition = fp500_15_p5 + fp1_00001_p5;
/// assert_eq!(addition, FixedP::<5>::from_units_frac(501, 15001)?);
/// # Ok(()) }
/// ```
///
/// The example below shows how you could use this type in a const context.
/// Note that the ergonomics are not great _currently_ due to lacking the
/// ability to panic in such contexts, not being able to use `?`, and in
/// general needing to call fallible functions. _However_, there are ways
/// to force compilation to fail on unexpected assumptions - this crate
/// (ab)uses those to hold its own guarantees:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use const_fixedp::FixedP;
/// const ONE: FixedP<4> = match FixedP::from_units(1) {
///     Ok(one) => one,
/// /// _ => unreachable!(), // not yet available on stable
/// #   _ => FixedP::zero(), // workaround for panicking on stable
/// };
/// const TEN_N_HALF: FixedP<4> = match FixedP::from_units_frac(10, 5000) {
///     Ok(n) => n,
/// /// _ => unreachable!(), // not yet available on stable
/// #   _ => FixedP::zero(), // workaround for panicking on stable
/// };
/// const ELEVEN_N_HALF: FixedP<4> = match ONE.checked_add(TEN_N_HALF) {
///     Some(n) => n,
/// /// _ => unreachable!(), // not yet available on stable
/// #   _ => FixedP::zero(), // workaround for panicking on stable
/// };
/// # Ok(()) }
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedP<const P: Precision> {
    // The inner u64 will encode the integer part in the upper bits
    // and the fractional part in the lower bits.
    // Note that we can't currently define a default representation
    // for this internal type because Rust does not yet support
    // specifying a default generic type in addition to constant type
    // parameters.
    n: u64,
}

impl<const P: Precision> FixedP<P> {
    /// Minimum number of integers that should be representable, arbitrarily set to 100 billion.
    // We might want to have this as a const type generic parameter once default values can be
    // used -- otherwise the type becomes a bit too verbose.
    pub const MIN_UNIT_SCALE: u64 = 100_000_000_000;
    // Compute bits required to represent an arbitrary "huge unit value" MIN_UNIT_SCALE.
    const MIN_UNIT_SCALE_BITS: u32 =
        (const_option_unwrap!(Self::MIN_UNIT_SCALE.checked_next_power_of_two()) - 1).count_ones();

    // Full scale of the fractional part.
    const FRAC_SCALE: u64 = { 10u64.pow(const_int_as!(P, Precision, u32)) };
    // Bitmask for the bits representing the fractional part, must be contiguous
    const BITMASK_FRAC: u64 =
        const_option_unwrap!(Self::FRAC_SCALE.checked_next_power_of_two()) - 1;
    // Resolution for the fractional part.
    const FRAC_BITS: u32 = Self::BITMASK_FRAC.count_ones();

    // Bitmask for the bits representing the integer part, must be contiguous
    const BITMASK_UNIT: u64 = !Self::BITMASK_FRAC;

    // Maximum fractional value
    const FRAC_MAX: u64 = const_option_unwrap!(Self::FRAC_SCALE.checked_sub(1));
    // Maximum integer value
    const UNIT_MAX: u64 = const_option_unwrap!(Self::BITMASK_UNIT.checked_shr(Self::FRAC_BITS));
    // Maximum value in both integer and fractional parts
    const MAX_VALUE: u64 = Self::BITMASK_UNIT | Self::FRAC_MAX;

    // Resolution for the integer part.
    // This ensures the type has enough bit space to hold a "huge unit value".
    const UNIT_BITS: u32 = {
        let unit_bits = const_option_unwrap!(u64::BITS.checked_sub(Self::FRAC_BITS));
        if unit_bits >= Self::MIN_UNIT_SCALE_BITS {
            unit_bits
        } else {
            const_panic!()
        }
    };
    // Mask to check for unit values that overflow the available unit bit space.
    const UNIT_OVERFLOW_MASK: u64 =
        const_option_unwrap!(Self::BITMASK_FRAC.checked_shl(Self::UNIT_BITS));

    /// Returns the precision for the fractional part of this fixed point type.
    pub const fn precision() -> Precision {
        P
    }

    /// Returns the bits taken for the integer part of this fixed point type.
    pub const fn unit_bits() -> u32 {
        Self::UNIT_BITS
    }

    /// Returns the bits taken for the fractional part of this fixed point type.
    pub const fn frac_bits() -> u32 {
        Self::FRAC_BITS
    }

    /// Returns the maximum integer representable by this fixed point type.
    pub const fn unit_max() -> u64 {
        Self::UNIT_MAX
    }

    /// Returns the maximum fractional part representable by this fixed point type.
    pub const fn frac_max() -> u64 {
        Self::FRAC_MAX
    }

    /// Returns a fixed point value representing the number zero.
    pub const fn zero() -> Self {
        // Can't use Self::from_units as it will require unwrapping/panicking
        // in a const fn, which is unstable at the moment.
        Self { n: 0 }
    }

    /// Returns the minimum value for this fixed point type.
    pub const fn min() -> Self {
        Self::zero()
    }

    /// Returns the maximum value for this fixed point type.
    pub const fn max() -> Self {
        Self { n: Self::MAX_VALUE }
    }

    /// Constructs a fixed point value from separate integer and fractional parts.
    pub const fn from_units_frac(units: u64, frac: u64) -> Result<Self, Error<P>> {
        if units & Self::UNIT_OVERFLOW_MASK > 0 {
            return Err(Error::OutOfRangeUnits);
        }
        if frac > Self::FRAC_MAX {
            return Err(Error::OutOfRangeFrac);
        }
        Ok(Self {
            n: (units << Self::FRAC_BITS) | frac,
        })
    }

    /// Constructs a fixed point value from an integer and a fractional part of 0.
    pub const fn from_units(units: u64) -> Result<Self, Error<P>> {
        Self::from_units_frac(units, 0)
    }

    /// Returns the integer part of a fixed point value.
    pub const fn units(&self) -> u64 {
        self.n >> Self::FRAC_BITS
    }

    /// Returns the fractional part of a fixed point value.
    pub const fn frac(&self) -> u64 {
        self.n & Self::BITMASK_FRAC
    }

    /// Returns the integer and fractional as a tuple of (units, frac).
    pub const fn split(&self) -> (u64, u64) {
        (self.units(), self.frac())
    }

    /// Checked integer addition. Computes self + rhs, returning None if overflow occurred.
    ///
    /// ```
    /// # fn main() -> Result<(), const_fixedp::Error<2>> {
    /// # use const_fixedp::FixedP;
    /// /// Create 1.95 and 1.05 as fixed point values
    /// let fixed_point_1_95 = FixedP::<2>::from_units_frac(1, 95)?;
    /// let fixed_point_1_05 = FixedP::<2>::from_units_frac(1, 5)?;
    ///
    /// /// Assert that the addition equals 3.00
    /// let addition = fixed_point_1_95 + fixed_point_1_05;
    /// assert_eq!(addition.units(), 3);
    /// assert_eq!(addition.frac(), 0);
    /// assert_eq!(&format!("{}", addition), "3.00");
    /// # Ok(()) }
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        // There must be at least one upper bit reserved for fractions when
        // calling units(), since const evalution will fail otherwise.
        // With that premise this addition can never overflow its unsigned
        // type, though it could overflow its assigned bit space.
        // ie. 0111 + 0111 = 1110
        let mut units = self.units() + rhs.units();

        // Compute the fractional addition and carry over.
        let mut frac = self.frac() + rhs.frac();

        if frac > Self::FRAC_MAX {
            frac -= Self::FRAC_SCALE;
            units += 1; // CO can't possibly overflow the unsigned integer type
                        // though bit space overflow will be caught below.
        }
        // Check for assigned bit space overflow.
        if units & Self::UNIT_OVERFLOW_MASK > 0 {
            return None;
        }

        Some(Self {
            n: (units << Self::FRAC_BITS) | frac,
        })
    }

    /// Checked integer subtraction. Computes self - rhs, returning None if underflow occurred.
    ///
    /// ```
    /// # fn main() -> Result<(), const_fixedp::Error<3>> {
    /// # use const_fixedp::FixedP;
    /// /// Create 11.50 and 1.05 as fixed point values
    /// let fixed_point_11_5 = FixedP::<3>::from_units_frac(11, 500)?;
    /// let fixed_point_1_05 = FixedP::<3>::from_units_frac(1, 50)?;
    ///
    /// /// Assert that the subtraction equals 10.45
    /// let subtraction = fixed_point_11_5 - fixed_point_1_05;
    /// assert_eq!(subtraction.units(), 10);
    /// assert_eq!(subtraction.frac(), 450);
    /// assert_eq!(&format!("{}", subtraction), "10.450");
    /// # Ok(()) }
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        let mut units = match self.units().checked_sub(rhs.units()) {
            Some(units) => units,
            _ => return None,
        };

        // There must be at least one upper bit reserved for units when
        // calling frac(), since const evaluation will fail otherwise.
        // With that premise we can cast the calls to frac() to a (positive)
        // signed type, and the subtraction can never underflow.
        // ie. 0000 - 0111 = 1111
        //
        // Compute the fractional subtraction and carry over.
        let mut frac = self.frac() as i64 - rhs.frac() as i64;

        if frac.is_negative() {
            frac += Self::FRAC_SCALE as i64;
            // CO can underflow.
            units = match units.checked_sub(1) {
                Some(units) => units,
                _ => return None,
            };
        }

        Some(Self {
            n: (units << Self::FRAC_BITS) | frac as u64,
        })
    }

    /// This method will convert (whenever possible) a `FixedP<Q>` to
    /// a `FixedP<P>`.
    ///
    /// Such a conversion is only possible in the below circumstances:
    ///
    /// | Q to P relation | Integer value  | Fraction value        |
    /// |-----------------|----------------|-----------------------|
    /// | Equal           | Indifferent    | Indifferent           |
    /// | Smaller         | Smaller scale  | Indifferent           |
    /// | Bigger          | Indifferent    | Q-P last digits are 0 |
    ///
    /// Essentially the conversion will be possible whenever a loss of
    /// precision in the fractional part or loss of scale in the integer
    /// part can still accomodate the same exact number.
    ///
    /// So a [`FixedP`] with a precision equal to another one can always
    /// be converted, but one that has higher precision can only be
    /// converted if the extra precision in the fractional part is not
    /// really used, that is, the extra digits are not significant, ie. 0,
    /// and one with lower precision can only be converted if the integer
    /// value fits in the reduced scale of the higher precision one.
    ///
    /// **Note**: currently Rust won't allow us to write the [`TryFrom`]
    /// implementation because of limitations in the places where const
    /// generic types can appear, but more importantly, because there is a
    /// blanket `impl<U, T> TryFrom<U> for T where U: Into<T>` that will
    /// conflict with the `impl<const Q, const P> TryFrom<FixedP<Q>> for FixedP<P>`
    /// _when P == Q_.
    ///
    /// This means we can't have [`TryFrom`] implemented for different
    /// precision values for the time being, but we can still implement
    /// this inherent method that does exactly that.
    pub const fn try_from_fixed<const Q: Precision>(value: FixedP<Q>) -> Result<Self, Error<P>> {
        let diff: i16 = P as i16 - Q as i16;
        let exp = diff.abs() as u32;

        let (units, frac) = if diff.is_positive() {
            // We can convert this fixed point value by just multiplying the fractional
            // part by 10^diff.
            //
            // Note: cannot use `?` in const fns as of 1.54.0.
            let factor = match 10u64.checked_pow(exp) {
                Some(factor) => factor,
                None => return Err(Error::OutOfRangeFrac),
            };
            let frac = match value.frac().checked_mul(factor) {
                Some(frac) => frac,
                None => return Err(Error::OutOfRangeFrac),
            };

            (value.units(), frac)
        } else if diff.is_negative() {
            // We can only convert those cases in which we can lose precision while
            // maintaining the same semantics, that is, those where the lost precision
            // digits would be zero.
            let mut frac = value.frac();
            if frac % exp as u64 == 0 {
                let divisor = match 10u64.checked_pow(exp) {
                    Some(divisor) => divisor,
                    None => return Err(Error::OutOfRangeFrac),
                };
                frac /= divisor;
            } else {
                return Err(Error::OutOfRangeFrac);
            }
            (value.units(), frac)
        } else {
            // No difference in precision, so no need to compute anything.
            // We could do just fine if we had the "safe transmute" feature
            // in stable, or otherwise resort to `unsafe` to cast `value` to
            // `FixedP<P>`, but we'll be 100% safe (and slightly suboptimal)
            // just calling the constructor.
            //
            // return Ok(unsafe { *(&value as *const _ as *const FixedP<P>) });
            (value.units(), value.frac())
        };

        Self::from_units_frac(units, frac)
    }
}

impl<const P: Precision> Default for FixedP<P> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const P: Precision> core::fmt::Display for FixedP<P> {
    /// The [`Display`][core::fmt::Display] implementation will display
    /// the integral part, followed by a dot `.` and the fractional part
    /// with the chosen precision, even if the fixed point value has a
    /// fractional part of zero.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}.{:0width$}",
            self.units(),
            self.frac(),
            width = (P as usize)
        )
    }
}

impl<const P: Precision> core::fmt::Debug for FixedP<P> {
    /// The [`Debug`][core::fmt::Debug] implementation for fixed point
    /// values will show the allocated bits for the integral and
    /// fractional parts as well as the maximum values for each and the
    /// internal representation.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("FixedP<{}>", P))
            .field("units", &self.units())
            .field("frac", &self.frac())
            .field("unit_bits", &Self::UNIT_BITS)
            .field("frac_bits", &Self::FRAC_BITS)
            .field("unit_max", &Self::UNIT_MAX)
            .field("frac_max", &Self::FRAC_MAX)
            .field("bitvalue", &self.n)
            .finish()
    }
}

impl<const P: Precision> core::str::FromStr for FixedP<P> {
    type Err = Error<P>;

    /// This function will parse a fixed point value out of a string slice.
    /// An error will be returned if the string cannot be parsed as a fixed
    /// point number, or if the integral or the fractional part are not
    /// representable with the chosen decimal precision.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (units, frac) = match s.split_once('.') {
            Some((units_s, frac_s)) => {
                let mut frac = frac_s.parse()?;
                // frac_s can't possibly parse to a u64 and have len >= 256, so the
                // conversion to a u8 below is safe.
                if let Some(digits) = P.checked_sub(frac_s.len() as u8) {
                    if digits > 0 {
                        // can't possibly overflow a u64 because the bigger the parsed
                        // fractional part is, the smaller the power argument is, but
                        // more importantly, because maximum value is bounded by 10^P,
                        // which is statically guaranteed to fit a u64.
                        frac *= 10u64.pow(digits as u32);
                    }
                } else {
                    // If the frac string takes more than P digits it is out of range.
                    return Err(Error::OutOfRangeFrac);
                }
                (units_s.parse()?, frac)
            }
            None => (s.parse()?, 0),
        };

        Self::from_units_frac(units, frac)
    }
}

macro_rules! impl_integer_conversion {
    ( $($uint:ty),+ ) => {
        $(
            impl<const P: Precision> TryFrom<$uint> for FixedP<P> {
                type Error = Error<P>;

                fn try_from(value: $uint) -> Result<Self, Self::Error> {
                    Self::from_units(u64::try_from(value)?)
                }
            }
        )+
    };
}

impl_integer_conversion!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl<const P: Precision> core::ops::Add for FixedP<P> {
    type Output = FixedP<P>;

    /// Add two `FixedP`'s.
    ///
    /// # Panics
    ///
    /// This function panics if the addition would overflow the bit space
    /// reserved for the integer part.
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("overflow")
    }
}

impl<const P: Precision> core::ops::AddAssign for FixedP<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: Precision> core::ops::Sub for FixedP<P> {
    type Output = FixedP<P>;

    /// Subtract two `FixedP`'s.
    ///
    /// # Panics
    ///
    /// This function panics if the subtraction would result in a
    /// negative integer part.
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs).expect("underflow")
    }
}

impl<const P: Precision> core::ops::SubAssign for FixedP<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Default precision used for testing, unless otherwise specified.
    const P: Precision = 8;
    const ZERO: FixedP<P> = FixedP::zero();
    const MIN: FixedP<P> = FixedP::min();
    const MAX: FixedP<P> = FixedP::max();

    mod overflowing {
        use super::*;

        #[test]
        #[should_panic(expected = "overflow")]
        fn overflow_units() {
            let one = FixedP::from_units(1).unwrap();

            let _overflow = MAX + one;
        }

        #[test]
        #[should_panic(expected = "underflow")]
        fn underflow_units() {
            let one = FixedP::from_units(1).unwrap();

            let _underflow = MIN - one;
        }

        #[test]
        #[should_panic(expected = "overflow")]
        fn overflow_frac() {
            let one = FixedP::from_units_frac(0, 1).unwrap();

            let _overflow = MAX + one;
        }

        #[test]
        #[should_panic(expected = "underflow")]
        fn underflow_frac() {
            let one = FixedP::from_units_frac(0, 1).unwrap();

            let _underflow = MIN - one;
        }

        #[test]
        #[should_panic(expected = "overflow")]
        fn overflow_max() {
            let _overflow = MAX + MAX;
        }

        #[test]
        #[should_panic(expected = "underflow")]
        fn underflow_max() {
            let _underflow = MIN - MAX;
        }
    }

    mod arithmetic {
        use super::*;

        #[test]
        fn addition() -> Result<(), Error<P>> {
            let a = FixedP::from_units_frac(2, 1)?;
            let b = FixedP::from_units_frac(6, 4)?;

            let res = a + b;

            assert_eq!(8, res.units());
            assert_eq!(5, res.frac());

            Ok(())
        }

        #[test]
        fn addition_assignment() -> Result<(), Error<P>> {
            let mut a = FixedP::from_units_frac(2, 1)?;
            let b = FixedP::from_units_frac(6, 4)?;

            a += b;

            assert_eq!(8, a.units());
            assert_eq!(5, a.frac());

            Ok(())
        }

        #[test]
        fn subtraction() -> Result<(), Error<P>> {
            let a = FixedP::from_units_frac(5, 2)?;
            let b = FixedP::from_units_frac(2, 1)?;

            let res = a - b;

            assert_eq!(3, res.units());
            assert_eq!(1, res.frac());

            Ok(())
        }

        #[test]
        fn subtraction_assignment() -> Result<(), Error<P>> {
            let mut a = FixedP::from_units_frac(5, 2)?;
            let b = FixedP::from_units_frac(2, 1)?;

            a -= b;

            assert_eq!(3, a.units());
            assert_eq!(1, a.frac());

            Ok(())
        }

        #[test]
        fn addition_with_carry_over() -> Result<(), Error<P>> {
            let almost_three = FixedP::from_units_frac(2, FixedP::<P>::FRAC_MAX)?;
            let slightly_above_one = FixedP::from_units_frac(1, 2)?;

            let res = almost_three + slightly_above_one;

            assert_eq!(4, res.units());
            assert_eq!(1, res.frac());

            Ok(())
        }

        #[test]
        fn subtraction_with_carry_over() -> Result<(), Error<P>> {
            let just_above_two = FixedP::from_units_frac(2, 1)?;
            let slightly_above_one = FixedP::from_units_frac(1, 2)?;

            let res = just_above_two - slightly_above_one;

            let (units, frac) = res.split();

            assert_eq!(0, units);
            assert_eq!(FixedP::<P>::FRAC_MAX, frac);

            Ok(())
        }
    }

    mod order {
        use super::*;

        #[test]
        fn sorting() -> Result<(), Error<P>> {
            let one = FixedP::from_units(1)?;
            let ten = FixedP::from_units(10)?;
            let mut v = vec![one, MAX, ZERO, ten];

            v.sort();
            assert_eq!(v.as_slice(), &[ZERO, one, ten, MAX]);

            Ok(())
        }
    }

    mod constructors {
        use super::*;

        #[test]
        fn zero() -> Result<(), Error<P>> {
            let zero_from_units = FixedP::from_units(0)?;

            assert!(ZERO == zero_from_units);
            assert_eq!(ZERO, zero_from_units);

            Ok(())
        }

        #[test]
        fn default_is_zero() -> Result<(), Error<P>> {
            let zero_from_units = FixedP::from_units(0)?;

            assert_eq!(FixedP::<P>::default(), zero_from_units);

            Ok(())
        }

        #[test]
        fn from_units_out_of_range_units() -> Result<(), Error<P>> {
            let oor = FixedP::from_units(u64::MAX);

            assert!(oor.is_err());
            assert_eq!(oor.unwrap_err(), Error::<P>::OutOfRangeUnits);

            Ok(())
        }

        #[test]
        fn from_units_frac_out_of_range_frac() -> Result<(), Error<P>> {
            let oor = FixedP::from_units_frac(0, FixedP::<P>::FRAC_SCALE);

            assert!(oor.is_err());
            assert_eq!(oor.unwrap_err(), Error::<P>::OutOfRangeFrac);

            Ok(())
        }

        #[test]
        fn try_from_conversion_error() -> Result<(), Error<P>> {
            let converted = FixedP::try_from(u128::MAX);

            assert!(matches!(converted, Err(Error::<P>::ConversionError(_))));

            Ok(())
        }

        mod try_from_fixed {
            use super::*;

            // Convert from a smaller to a bigger precision fixed point value.
            #[test]
            fn smaller() -> Result<(), Box<dyn std::error::Error>> {
                let two_point_one = FixedP::<1>::from_units_frac(2, 1)?;
                let two_point_one_hundred = FixedP::<3>::try_from_fixed(two_point_one);

                assert!(two_point_one_hundred.is_ok());
                assert_eq!(
                    two_point_one_hundred.unwrap(),
                    FixedP::<3>::from_units_frac(2, 100)?
                );

                Ok(())
            }

            // Maximum integers of smaller precision types are not representable by
            // bigger precision fixed point types.
            #[test]
            fn smaller_max_units() -> Result<(), Box<dyn std::error::Error>> {
                let max_units_one_decimal = FixedP::<1>::from_units(FixedP::<1>::UNIT_MAX)?;
                let three_decimal = FixedP::<3>::try_from_fixed(max_units_one_decimal);

                assert!(three_decimal.is_err());
                assert_eq!(three_decimal.unwrap_err(), Error::<3>::OutOfRangeUnits);

                Ok(())
            }

            // Convert from a bigger to a smaller precision fixed point value.
            #[test]
            fn bigger() -> Result<(), Box<dyn std::error::Error>> {
                let two_point_one_hundred = FixedP::<3>::from_units_frac(2, 100)?;
                let two_point_one = FixedP::<1>::try_from_fixed(two_point_one_hundred);

                assert!(two_point_one.is_ok());
                assert_eq!(two_point_one.unwrap(), FixedP::<1>::from_units_frac(2, 1)?);

                Ok(())
            }

            // The minimal fractional part (non-zero) of a bigger precision type is not
            // representable by bigger precision fixed point types.
            #[test]
            fn bigger_frac_one() -> Result<(), Box<dyn std::error::Error>> {
                let two_point_one_thousandth = FixedP::<3>::from_units_frac(2, 1)?;
                let one_decimal = FixedP::<1>::try_from_fixed(two_point_one_thousandth);

                assert!(one_decimal.is_err());
                assert_eq!(one_decimal.unwrap_err(), Error::<1>::OutOfRangeFrac);

                Ok(())
            }

            #[test]
            fn equal() -> Result<(), Box<dyn std::error::Error>> {
                let two_point_one_hundred = FixedP::<3>::from_units_frac(2, 100)?;
                let two_point_another_hundred = FixedP::<3>::try_from_fixed(two_point_one_hundred);

                assert!(two_point_another_hundred.is_ok());
                assert_eq!(two_point_another_hundred.unwrap(), two_point_one_hundred);

                Ok(())
            }
        }
    }

    mod representation {
        use super::*;

        #[test]
        fn display() -> Result<(), Error<P>> {
            let val = FixedP::from_units_frac(12, 3)?;

            assert_eq!(
                &format!("{}", val),
                &format!("12.{:0width$}", 3, width = (P as usize))
            );

            Ok(())
        }

        #[test]
        fn display_frac_zero() -> Result<(), Error<P>> {
            let val = FixedP::from_units(12)?;

            assert_eq!(
                &format!("{}", val),
                &format!("12.{:0width$}", 0, width = (P as usize))
            );

            Ok(())
        }

        #[test]
        fn debug() -> Result<(), Error<P>> {
            let val = FixedP::from_units_frac(12, 3)?;

            let debug_s = format!(
                "FixedP<{P}> {{ \
                units: {units}, \
                frac: {frac}, \
                unit_bits: {unit_bits}, \
                frac_bits: {frac_bits}, \
                unit_max: {unit_max}, \
                frac_max: {frac_max}, \
                bitvalue: {bitvalue} }}",
                P = P,
                units = &val.units(),
                frac = &val.frac(),
                unit_bits = &FixedP::<P>::UNIT_BITS,
                frac_bits = &FixedP::<P>::FRAC_BITS,
                unit_max = &FixedP::<P>::UNIT_MAX,
                frac_max = &FixedP::<P>::FRAC_MAX,
                bitvalue = &val.n
            );

            assert_eq!(&format!("{:?}", val), &debug_s);

            Ok(())
        }
    }

    mod parsing {
        use super::*;

        #[test]
        fn integers() -> Result<(), Error<P>> {
            let val = FixedP::from_units(12)?;

            assert_eq!("12".parse::<FixedP<P>>()?, val);
            assert_eq!("12.0".parse::<FixedP<P>>()?, val);

            Ok(())
        }

        // Note: some tests below require P > 1, so we will use a local
        //       precision constant.
        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn frac_missing_right_zeroes() {
            const P: Precision = 3;
            assert!(P > 1);
            let val = FixedP::from_units_frac(12, 3 * 10u64.pow((P - 1) as u32)).unwrap();

            assert_eq!("12.3".parse::<FixedP<P>>().unwrap(), val);
        }

        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn frac_including_left_zeroes() -> Result<(), Box<dyn std::error::Error>> {
            const P: Precision = 3;
            assert!(P > 1);
            let val = FixedP::<P>::from_units_frac(12, 3 * 10u64.pow((P - 1 - 1) as u32))?;
            let s = &format!("12.{:0width$}", 3, width = ((P - 1) as usize));

            assert_eq!(s.parse::<FixedP<P>>()?, val);

            Ok(())
        }

        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn frac_out_of_range() -> Result<(), Box<dyn std::error::Error>> {
            const P: Precision = 3;
            assert!(P > 1);
            let s = &format!("12.{:0width$}3", width = (P as usize));
            let parsed = s.parse::<FixedP<P>>();

            assert!(parsed.is_err());
            assert!(matches!(parsed.unwrap_err(), Error::OutOfRangeFrac));

            Ok(())
        }

        #[test]
        fn units_out_of_range() -> Result<(), Box<dyn std::error::Error>> {
            let s = &format!("{}", u64::MAX);
            let parsed = s.parse::<FixedP<P>>();

            assert!(parsed.is_err());
            assert!(matches!(parsed.unwrap_err(), Error::OutOfRangeUnits));

            Ok(())
        }
    }

    // These test for a few well known floating point issues as specified in
    // IEEE 754 that are problematic for certain applications.
    //
    // These are specially problematic for handling currency.
    //
    // Check out https://en.wikipedia.org/wiki/Machine_epsilon for details.
    mod ieee754_gotchas {
        use super::*;

        // Floating points as spec'ed in IEEE 754 already show surprising effects
        // when performing simple operations to the point equality is unreliable.
        #[test]
        #[allow(clippy::float_cmp)]
        fn no_basic_rounding_error() -> Result<(), Box<dyn std::error::Error>> {
            let zero_20 = FixedP::<2>::from_units_frac(0, 20)?;
            let zero_15 = FixedP::<2>::from_units_frac(0, 15)?;
            let zero_04 = FixedP::<2>::from_units_frac(0, 4)?;
            let zero_09 = FixedP::<2>::from_units_frac(0, 9)?;

            assert_ne!(0.20 - 0.15 + 0.04, 0.09);
            assert_eq!(zero_20 - zero_15 + zero_04, zero_09);

            Ok(())
        }

        // When the bit space is exhausted, rounding happens on IEEE 754 f32.
        #[test]
        #[allow(clippy::float_cmp)]
        fn no_edge_case_rounding_error_f32() -> Result<(), Box<dyn std::error::Error>> {
            let f: f32 = 16777217 as f32;
            assert_eq!(f, 16777217.0);
            assert_eq!(f, 16777216 as f32);
            assert_eq!(f, 16777216.0);

            let fixed = FixedP::<2>::from_units_frac(16777217, 0)?;
            assert_eq!(fixed.units(), 16777217);
            assert_eq!(fixed.frac(), 0);
            // For fixed point values there should be no rounding ever.
            assert_ne!(fixed.units(), 16777216);
            let parsed_f32 = format!("{}", fixed).parse::<f32>()?;
            assert_eq!(parsed_f32, 16777217.0);
            assert_ne!(format!("{}", parsed_f32).parse::<FixedP<2>>()?, fixed);

            Ok(())
        }

        // When the bit space is exhausted, rounding happens on IEEE 754 f64.
        #[test]
        #[allow(clippy::float_cmp)]
        fn no_edge_case_rounding_error_f64() -> Result<(), Box<dyn std::error::Error>> {
            let f: f64 = 9007199254740993u64 as f64;
            assert_eq!(f, 9007199254740993.0);
            assert_eq!(f, 9007199254740992u64 as f64);
            assert_eq!(f, 9007199254740992.0);

            let fixed = FixedP::<2>::from_units_frac(9007199254740993, 0)?;
            assert_eq!(fixed.units(), 9007199254740993);
            assert_eq!(fixed.frac(), 0);
            // For fixed point values there should be no rounding ever.
            assert_ne!(fixed.units(), 9007199254740992);
            let parsed_f64 = format!("{}", fixed).parse::<f64>()?;
            assert_eq!(parsed_f64, 9007199254740993.0);
            assert_ne!(format!("{}", parsed_f64).parse::<FixedP<2>>()?, fixed);

            Ok(())
        }

        // Finally let's assert that fixed point values don't do rounding at max value.
        #[test]
        fn no_max_value_rounding_error() -> Result<(), Box<dyn std::error::Error>> {
            let fixed = FixedP::<2>::from_units_frac(FixedP::<2>::UNIT_MAX, 0)?;
            assert_eq!(fixed.units(), FixedP::<2>::UNIT_MAX);
            assert_eq!(fixed.frac(), 0);
            assert_ne!(fixed.units(), FixedP::<2>::UNIT_MAX - 1);

            let fixed = FixedP::<2>::from_units_frac(FixedP::<2>::UNIT_MAX, FixedP::<2>::FRAC_MAX)?;
            assert_eq!(fixed.units(), FixedP::<2>::UNIT_MAX);
            assert_eq!(fixed.frac(), FixedP::<2>::FRAC_MAX);
            // For fixed point values there should be no rounding ever.
            assert_ne!(fixed.units(), FixedP::<2>::UNIT_MAX - 1);
            assert_ne!(fixed.frac(), FixedP::<2>::FRAC_MAX - 1);

            Ok(())
        }
    }
}
