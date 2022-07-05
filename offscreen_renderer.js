import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

export async function sharedWorker2() {
    const worker = new SharedWorker("worker.js", { type: "module" });
    /**
     * SharedWorkers communicate via the `postMessage` function in their `port` property.
     * Therefore you must use the SharedWorker's `port` property when calling `Comlink.wrap`.
     */
    const obj = Comlink.wrap(worker.port);
    console.log(`Counter please: ${await obj.counter}`);
    await obj.inc();
    console.log(`Counter pretty please: ${await obj.counter}`);
}
