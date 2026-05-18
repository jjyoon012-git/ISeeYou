(function (global) {
  function levelClass(level) {
    if (level === 'high') return 'level-high';
    if (level === 'uncertain') return 'level-uncertain';
    return 'level-low';
  }

  function labelText(disclosure) {
    const pct = Number.isFinite(disclosure.percent) ? ` ${disclosure.percent}%` : '';
    if (disclosure.level === 'high') return `AI 생성 가능성 높음${pct}`;
    if (disclosure.level === 'uncertain') return `AI 생성 여부 확인 필요${pct}`;
    return `AI 생성 가능성 낮음${pct}`;
  }

  global.ISY_DEMO = { levelClass, labelText };
})(window);
