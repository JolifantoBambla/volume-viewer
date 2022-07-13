import { openGroup, openArray, slice } from 'https://cdn.skypack.dev/zarr';

self.onmessage = e => {
    console.log(e);
    openGroup(e.data.store, e.data.path)
        .then(group => {
            self.postMessage(group);
        })
};
