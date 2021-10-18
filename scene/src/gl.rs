mod generated {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use generated::*;
include!(concat!(env!("OUT_DIR"), "/safe_wrapper.rs"));

pub use generated::__gl_imports::raw::c_int as int;
pub use generated::__gl_imports::raw::c_uint as uint;
