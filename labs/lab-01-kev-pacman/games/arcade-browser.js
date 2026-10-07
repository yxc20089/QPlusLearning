// Colab player controller for the unchanged upstream game. GPL-3.0.
(function () {
    var mode = 'human', running = false, waiting = false, needDecision = true;
    var generation = 0, requestId = 0, pending = {}, last = 0, accumulator = 0, lastStatus = 0;
    var moved = false, startPixel = null, actionFrames = 0;
    function post(type, data) { parent.postMessage({channel: 'kev-classic-pacman', type: type, data: data}, '*'); }
    function rpc(callback, argument) {
        return new Promise(function(resolve, reject) {
            var id = ++requestId;
            var timer = setTimeout(function() { delete pending[id]; reject(new Error('Notebook callback timed out after 90 seconds.')); }, 90000);
            pending[id] = {resolve: resolve, reject: reject, timer: timer};
            parent.postMessage({channel: 'kev-classic-pacman', type: 'callback', id: id,
                callback: callback, argument: argument}, '*');
        });
    }
    function fail(error) { running = false; waiting = false; post('error', String(error.message || error)); }
    async function decide() {
        if (waiting) return;
        waiting = true;
        var selectedGeneration = generation;
        post('status', {message: 'Kev is choosing; simulation clock paused.'});
        try {
            var response = await rpc('pacman.decide', labArcade.observe());
            if (generation !== selectedGeneration) return;
            var answer = response.answers.move;
            var legal = labArcade.legalMoves();
            if (legal.indexOf(answer.choice) < 0) throw new Error('The model returned an illegal direction.');
            var total = legal.reduce(function(sum, key) {
                var p = answer.probabilities[key];
                if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error('Invalid player probabilities.');
                return sum + p;
            }, 0);
            if (Math.abs(total - 1) > .002) throw new Error('Player probabilities do not sum to one.');
            if (!response.active_checkpoint) throw new Error('The active adapter was not verified.');
            post('decision', response);
            labArcade.setMove(answer.choice);
            startPixel = {x: pacman.pixel.x, y: pacman.pixel.y};
            moved = false; actionFrames = 0; needDecision = false; waiting = false;
        } catch (error) { if (generation === selectedGeneration) fail(error); }
    }
    window.addEventListener('message', async function(event) {
        if (event.source !== parent || !event.data || event.data.channel !== 'kev-classic-pacman') return;
        var message = event.data;
        if (message.type === 'callback-result') {
            var p = pending[message.id];
            if (!p) return;
            clearTimeout(p.timer); delete pending[message.id];
            if (message.error) p.reject(new Error(message.error)); else p.resolve(message.data);
            return;
        }
        if (message.type !== 'command') return;
        var cmd = message.data;
        if (cmd === 'pause') { generation++; running = false; waiting = false; needDecision = true; audio.silence(); }
        if (cmd === 'reset') {
            generation++; running = false; waiting = false; needDecision = true;
            audio.silence(); labArcade.reset({seed: 7, action_version: globalThis.LAB_ACTION_VERSION || 1}); labArcade.draw();
        }
        if (cmd === 'human' || cmd === 'kev') {
            generation++; mode = cmd; waiting = false; needDecision = true; running = false;
            pacman.clearInputDir(); audio.silence();
        }
        if (cmd === 'start') {
            var selectedGeneration = ++generation;
            try {
                if (mode === 'kev') post('identity', await rpc('pacman.model', null));
                if (selectedGeneration === generation) { running = true; waiting = false; needDecision = true; }
            } catch (error) { if (selectedGeneration === generation) fail(error); }
        }
        post('status', {message: running ? 'Playing — ' + mode : 'Paused — ' + mode});
    });
    var keys = {ArrowUp: 0, w: 0, W: 0, ArrowLeft: 1, a: 1, A: 1,
                ArrowDown: 2, s: 2, S: 2, ArrowRight: 3, d: 3, D: 3};
    ['keydown', 'keyup'].forEach(function(type) {
        window.addEventListener(type, function(event) {
            event.stopImmediatePropagation(); event.preventDefault();
            if (mode === 'human' && keys[event.key] !== undefined) {
                if (type === 'keydown') pacman.setInputDir(keys[event.key]); else pacman.clearInputDir(keys[event.key]);
            }
        }, true);
    });
    ['touchstart', 'touchend', 'touchmove', 'click'].forEach(function(type) {
        window.addEventListener(type, function(event) {
            if (mode === 'kev' || type === 'click') event.stopImmediatePropagation();
        }, true);
    });
    function frame(now) {
        accumulator += Math.min(50, now - (last || now)); last = now;
        if (!running || waiting) accumulator = 0;
        while (running && !waiting && accumulator >= 1000 / 60) {
            accumulator -= 1000 / 60;
            if (state === overState) { running = false; audio.silence(); post('status', {message: 'Game over — reset to play again.'}); break; }
            if (mode === 'kev' && labArcade.playing() && needDecision) { decide(); accumulator = 0; break; }
            labArcade.tick();
            if (mode === 'kev' && !needDecision) {
                actionFrames++;
                moved = moved || pacman.pixel.x !== startPixel.x || pacman.pixel.y !== startPixel.y;
                if (labArcade.outcome() || labArcade.actionDone(moved)) {
                    labArcade.finishMove(); needDecision = true;
                } else if (actionFrames >= (globalThis.LAB_ACTION_VERSION === 2 ? 600 : 180)) { fail(new Error('Player action did not reach its declared tile boundary.')); }
            }
        }
        labArcade.draw();
        if (now - lastStatus > 250) {
            var observation = labArcade.observe();
            post('board', {score: observation.score, lives: observation.lives, level: observation.level,
                pellets: observation.pellets_remaining, phase: observation.ghost_phase,
                visits: observation.visits_to_current_tile, turn: observation.turn,
                simulation_frames: observation.simulation_frames, ghosts: observation.ghosts});
            lastStatus = now;
        }
        requestAnimationFrame(frame);
    }
    window.addEventListener('load', function() {
        executive.stop();
        // Upstream focus handlers call start; our clock owns all subsequent updates.
        executive.start = function() {};
        labArcade.reset({seed: 7, action_version: globalThis.LAB_ACTION_VERSION || 1});
        post('ready', null); requestAnimationFrame(frame);
    });
})();
