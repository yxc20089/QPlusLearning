// CPU-only scout sidecar. Uses the pinned engine's existing rewind API. GPL-3.0.
// It never exposes or changes the native future RNG and never controls a player
// during evaluation. Candidate labels require cold native continuation replay.
const fs = require('node:fs'), vm = require('node:vm'), readline = require('node:readline');
function Audio() { this.load = this.pause = this.addEventListener = this.removeEventListener = function () {}; this.play = function () { return Promise.resolve(); }; }
const context = {console: {log() {}, error() {}}, Audio, localStorage: {},
    window: {addEventListener() {}, location: {hash: ''}}, document: {},
    setTimeout() {}, clearTimeout() {}, setInterval() {}, clearInterval() {},
    requestAnimationFrame() {}, cancelAnimationFrame() {}};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
context.labArcade.headless();
const snapshots = new Map(); let counter = 0;
readline.createInterface({input: process.stdin}).on('line', line => {
    try {
        const input = JSON.parse(line), api = context.labArcade, bench = context.labBenchmark;
        let result;
        if (input.command === 'reset') {
            snapshots.clear(); counter = 0;
            result = api.reset({...input.options, skipReady: true});
        }
        else if (input.command === 'observe') result = api.observe();
        else if (input.command === 'step') result = bench.transition(input.direction);
        else if (input.command === 'risks') result = bench.risks();
        else if (input.command === 'continue') result = bench.continueLife();
        else if (input.command === 'snapshot') {
            const handle = ++counter;
            // Disjoint from the unchanged teacher and diagnostic rewind slots.
            snapshots.set(handle, api.searchSave(-200000000 - handle));
            result = {handle};
        }
        else if (input.command === 'restore') {
            if (!snapshots.has(input.handle)) throw new Error('Unknown native scout snapshot');
            api.searchRestore(snapshots.get(input.handle)); result = api.observe();
        }
        else if (input.command === 'forget') { snapshots.delete(input.handle); result = {forgotten: input.handle}; }
        else throw new Error('Unknown scout command');
        process.stdout.write(JSON.stringify({result}) + '\n');
    } catch (error) { process.stdout.write(JSON.stringify({error: String(error.stack || error)}) + '\n'); }
});
