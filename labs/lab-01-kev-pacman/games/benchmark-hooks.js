// Diagnostic-only integration inside the pinned game's closure. GPL-3.0.
// The v1 player observation, action boundaries and native physics stay unchanged.
globalThis.labBenchmark = (function () {
    // Native menus register listeners even in CPU-only terminal transitions.
    canvas = {addEventListener: function () {}, removeEventListener: function () {}};
    function transition(direction) {
        var before = labArcade.observe(), ghostEats = [], fruitEats = 0;
        var originals = ghosts.map(function (g) { return g.onEaten; });
        var testCollide = fruit.testCollide;
        ghosts.forEach(function (g, i) {
            g.onEaten = function () { ghostEats.push(g.name); return originals[i].apply(g, arguments); };
        });
        fruit.testCollide = function () {
            var score = getScore(), result = testCollide.apply(fruit, arguments);
            if (getScore() > score) fruitEats++;
            return result;
        };
        var result;
        try { result = labArcade.step(direction); }
        finally {
            ghosts.forEach(function (g, i) { g.onEaten = originals[i]; });
            fruit.testCollide = testCollide;
        }
        var normal = 0, power = 0;
        before.maze.forEach(function (row, y) {
            for (var x = 0; x < row.length; x++) {
                if (row[x] !== result.state.maze[y][x]) {
                    if (row[x] === '.') normal++;
                    if (row[x] === 'o') power++;
                }
            }
        });
        result.events = {normal_pellets: normal, power_pellets: power,
            ghosts_eaten: ghostEats, fruit_eaten: fruitEats};
        return result;
    }
    function risks() {
        var root = labArcade.searchSave(-900001), result = {};
        try {
            labArcade.legalMoves().forEach(function (move) {
                labArcade.searchRestore(root);
                var step = transition(move);
                result[move] = {life_lost: step.outcome === 'life_lost',
                    pellets: step.events.normal_pellets + step.events.power_pellets,
                    action_frames: step.action_frames, player: step.state.player};
            });
        } finally { labArcade.searchRestore(root); }
        return result;
    }
    function continueLife() {
        if (labArcade.outcome() === 'level_cleared') return {status: 'level_cleared', state: labArcade.observe()};
        var ticks = 0;
        while (state !== playState && state !== overState && ticks < 1000) { labArcade.tick(); ticks++; }
        if (state !== playState && state !== overState) throw new Error('Native life transition did not complete.');
        return {status: state === overState ? 'game_over' : 'playing', state: labArcade.observe(), animation_frames: ticks};
    }
    return {transition: transition, risks: risks, continueLife: continueLife};
})();
