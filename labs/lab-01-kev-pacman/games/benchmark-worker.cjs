// CPU-only scoring harness, native engine and v1 controller unchanged. GPL-3.0.
const fs = require('node:fs'), vm = require('node:vm'), readline = require('node:readline');
function Audio() { this.load = this.pause = this.addEventListener = this.removeEventListener = function () {}; this.play = function () { return Promise.resolve(); }; }
const context = {console: {log() {}, error() {}}, Audio, localStorage: {},
    window: {addEventListener() {}, location: {hash: ''}}, document: {},
    setTimeout() {}, clearTimeout() {}, setInterval() {}, clearInterval() {},
    requestAnimationFrame() {}, cancelAnimationFrame() {}};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
context.labArcade.headless();
let plan;
readline.createInterface({input: process.stdin}).on('line', line => {
    try {
        const input = JSON.parse(line), api = context.labArcade, bench = context.labBenchmark;
        let result;
        if (input.command === 'reset') result = api.reset({...input.options, skipReady: true});
        else if (input.command === 'observe') result = api.observe();
        else if (input.command === 'step') result = bench.transition(input.direction);
        else if (input.command === 'risks') result = bench.risks();
        else if (input.command === 'continue') result = bench.continueLife();
        else if (input.command === 'plan') {
            if (!plan) {
                vm.runInContext(fs.readFileSync(require('node:path').join(__dirname, 'arcade-planner.js'), 'utf8'), context);
                plan = context.createArcadePlanner(api);
            }
            result = plan(input.options);
        }
        else throw new Error('Unknown benchmark command');
        process.stdout.write(JSON.stringify({result}) + '\n');
    } catch (error) { process.stdout.write(JSON.stringify({error: String(error.stack || error)}) + '\n'); }
});
