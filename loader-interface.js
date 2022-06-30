import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

export async function sharedWorker() {
    const worker = new SharedWorker("worker.js");
    /**
     * SharedWorkers communicate via the `postMessage` function in their `port` property.
     * Therefore you must use the SharedWorker's `port` property when calling `Comlink.wrap`.
     */
    const obj = Comlink.wrap(worker.port);
    alert(`Counter: ${await obj.counter}`);
    await obj.inc();
    alert(`Counter: ${await obj.counter}`);
}
