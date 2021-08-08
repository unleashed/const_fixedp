//! This module provides a few macros to produce compiler errors when
//! assertions fail in constant contexts. Rust stable currently cannot
//! panic or call panicking functions in these contexts, so this module
//! enables a user to force a compilation error when a specific check
//! isn't met.
//!
//! There are helpers to deal with Result<T, E> and Option<T>, as well
//! as to emit assertions or directly cause a compilation error.
//!
//! This is useful, for example, to ensure some contant evaluation turns
//! out to be within a expected range of values.

/// Signal a fatal error when evaluating a constant context.
///
/// Can be passed an expression for type checking.
///
/// Note: Rust does not currently support panicking in const contexts in
///       stable versions, so this macro will just exhaust the evaluation
///       limit to force a compiler error.
#[macro_export(local_inner_macros)]
macro_rules! const_panic {
    ($default:expr) => {{
        let constant_check_has_panicked_ignore_eval_limit = true;
        while constant_check_has_panicked_ignore_eval_limit {}
        $default
    }};
    () => {
        const_panic!(0)
    };
}

/// Unwrap a Result<T, E> in constant contexts. Signal a fatal error on Err.
///
/// Can be passed an expression for type checking.
#[macro_export(local_inner_macros)]
macro_rules! const_result_unwrap {
    ($e:expr, $default:expr) => {
        match $e {
            Ok(v) => v,
            error => const_panic!($default),
        }
    };
    ($e:expr) => {
        const_result_unwrap!($e, 0)
    };
}

/// Unwrap an Option<T> in constant contexts. Signal a fatal error on None.
///
/// Can be passed an expression for type checking.
#[macro_export(local_inner_macros)]
macro_rules! const_option_unwrap {
    ($e:expr, $default:expr) => {
        match $e {
            Some(v) => v,
            _ => const_panic!($default),
        }
    };
    ($e:expr) => {
        const_option_unwrap!($e, 0)
    };
}

/// Assert a boolean condition in constant contexts. Signal a fatal error on failure.
#[macro_export(local_inner_macros)]
macro_rules! const_assert {
    ($e:expr) => {
        match $e {
            true => true,
            false => const_panic!(false),
        }
    };
}

/// Cast a given integer type to another one in constant contexts.
/// Signal a fatal error if the cast target can't fit the source integer type.
#[macro_export(local_inner_macros)]
macro_rules! const_int_as {
    ($e:expr, $lhs:ty, $rhs:ty) => {
        if core::mem::size_of::<$lhs>() <= core::mem::size_of::<$rhs>() {
            $e as $rhs
        } else {
            const_panic!(0) as $rhs
        }
    };
}
