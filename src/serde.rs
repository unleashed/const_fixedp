//! This module implements the [`Deserialize`] trait for the [`FixedP`] type.
//!
use serde::{de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

use super::{FixedP, Precision};

impl<const P: Precision> Serialize for FixedP<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{}", self))
    }
}

impl<'de, const P: Precision> Deserialize<'de> for FixedP<P> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FixedPVisitor<const P: Precision>;

        impl<'de, const P: Precision> Visitor<'de> for FixedPVisitor<P> {
            type Value = FixedP<P>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str(&format!("struct FixedP<{}>", P))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                use core::str::FromStr;
                Self::Value::from_str(v).map_err(|e| {
                    serde::de::Error::custom(&format!(
                        "unexpected string {}, expected FixedP<{}>: {}",
                        v, P, e
                    ))
                })
            }
        }

        deserializer.deserialize_str(FixedPVisitor)
    }
}

#[cfg(test)]
mod test {
    use super::{FixedP, Precision};

    mod serialize {
        use super::*;

        #[test]
        fn with_frac() -> Result<(), Box<dyn std::error::Error>> {
            let s = r#""12.030""#;
            let val = FixedP::<3>::from_units_frac(12, 30)?;
            let json = serde_json::to_string(&val);

            assert!(json.is_ok());
            assert_eq!(json.unwrap(), s);

            Ok(())
        }

        #[test]
        fn integer_only() -> Result<(), Box<dyn std::error::Error>> {
            let s = r#""12.000""#;
            let val = FixedP::<3>::from_units(12)?;
            let json = serde_json::to_string(&val);

            assert!(json.is_ok());
            assert_eq!(json.unwrap(), s);

            Ok(())
        }
    }

    mod deserialize {
        use super::*;

        #[test]
        fn with_frac() -> Result<(), Box<dyn std::error::Error>> {
            let s = r#""12.03""#;
            let val = FixedP::<3>::from_units_frac(12, 30)?;
            let v = serde_json::from_str::<FixedP<3>>(s);

            assert!(v.is_ok());
            assert_eq!(v.unwrap(), val);

            Ok(())
        }

        #[test]
        fn integer_only() -> Result<(), Box<dyn std::error::Error>> {
            let s = r#""12""#;
            let val = FixedP::<3>::from_units(12)?;
            let v = serde_json::from_str::<FixedP<3>>(s);

            assert!(v.is_ok());
            assert_eq!(v.unwrap(), val);

            Ok(())
        }

        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn frac_out_of_range() -> Result<(), Box<dyn std::error::Error>> {
            const OVER_1: Precision = 3;
            assert!(OVER_1 > 1);

            let s = format!(r#""12.{:0width$}""#, 5, width = ((OVER_1 + 1) as usize));
            let v = serde_json::from_str::<FixedP<3>>(&s);

            assert!(v.is_err());

            let expected_error_msg = format!(
                "unexpected string {}, expected FixedP<{}>: {}",
                s.replace('"', ""),
                OVER_1,
                crate::Error::<3>::OutOfRangeFrac
            );
            assert!(format!("{}", v.unwrap_err()).starts_with(&expected_error_msg));

            Ok(())
        }

        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn integer_out_of_range() -> Result<(), Box<dyn std::error::Error>> {
            const OVER_1: Precision = 3;
            assert!(OVER_1 > 1);

            let s = format!(r#""{}""#, u64::MAX);
            let v = serde_json::from_str::<FixedP<3>>(&s);

            assert!(v.is_err());

            let expected_error_msg = format!(
                "unexpected string {}, expected FixedP<{}>: {}",
                s.replace('"', ""),
                OVER_1,
                crate::Error::<OVER_1>::OutOfRangeUnits
            );
            assert!(format!("{}", v.unwrap_err()).starts_with(&expected_error_msg));

            Ok(())
        }

        #[test]
        #[allow(clippy::assertions_on_constants)]
        fn negative() -> Result<(), Box<dyn std::error::Error>> {
            const OVER_1: Precision = 3;
            assert!(OVER_1 > 1);

            let s = format!(r#""{}""#, i64::MIN);
            let v = serde_json::from_str::<FixedP<3>>(&s);

            assert!(v.is_err());

            // We can't instantiate the exact parsing error without actually
            // failing to parse something in a similar manner due to privacy
            // rules in the parsing error from rust's core::num library.
            //
            // The test below assumes that deserializing goes through the same
            // machinery as FromStr::from_str to result in the same error message.
            let parse_error = format!("{}", i64::MIN).parse::<FixedP<3>>();
            assert!(parse_error.is_err());

            let expected_error_msg = format!(
                "unexpected string {}, expected FixedP<{}>: {}",
                s.replace('"', ""),
                OVER_1,
                parse_error.unwrap_err()
            );

            assert!(format!("{}", v.unwrap_err()).starts_with(&expected_error_msg));

            Ok(())
        }
    }
}
