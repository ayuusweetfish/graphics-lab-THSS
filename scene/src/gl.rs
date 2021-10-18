mod generated {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use generated::*;
include!(concat!(env!("OUT_DIR"), "/safe_wrapper.rs"));
