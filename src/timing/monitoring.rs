use serde::Serialize;

#[derive(Copy, Clone, Debug, Default, Serialize)]
pub struct MonitoringDataFrame {
    pub dvr: f64,
    #[serde(rename = "dvrBegin")]
    pub dvr_begin: f64,
    #[serde(rename = "dvrEnd")]
    pub dvr_end: f64,

    pub present: f64,
    #[serde(rename = "presentBegin")]
    pub present_begin: f64,
    #[serde(rename = "presentEnd")]
    pub present_end: f64,

    #[serde(rename = "lruUpdate")]
    pub lru_update: f64,
    #[serde(rename = "lruUpdateBegin")]
    pub lru_update_begin: f64,
    #[serde(rename = "lruUpdateEnd")]
    pub lru_update_end: f64,

    #[serde(rename = "processRequests")]
    pub process_requests: f64,
    #[serde(rename = "processRequestsBegin")]
    pub process_requests_begin: f64,
    #[serde(rename = "processRequestsEnd")]
    pub process_requests_end: f64,

    #[serde(rename = "octreeUpdate")]
    pub octree_update: f64,
    #[serde(rename = "octreeUpdateBegin")]
    pub octree_update_begin: f64,
    #[serde(rename = "octreeUpdateEnd")]
    pub octree_update_end: f64,
}
