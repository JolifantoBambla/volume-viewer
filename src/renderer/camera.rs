pub trait Camera {}

pub struct OrthographicCamera {}

impl Camera for OrthographicCamera {}

pub struct PerspectiveCamera {}

impl Camera for PerspectiveCamera {}
