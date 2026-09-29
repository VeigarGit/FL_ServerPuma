const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('.site-nav');
function closeMenu() {
  nav?.classList.remove('is-open');
  menu?.setAttribute('aria-expanded', 'false');
}
menu?.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  menu.setAttribute('aria-expanded', String(open));
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && nav?.classList.contains('is-open')) {
    closeMenu();
    menu.focus();
  }
});
document.addEventListener('click', event => {
  if (!event.target.closest('.site-header')) closeMenu();
});
nav?.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));

document.querySelectorAll('.copy-button').forEach(button => {
  button.addEventListener('click', async () => {
    const code = button.parentElement.querySelector('code');
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied!';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Selected — copy manually';
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 2500);
  });
});

document.querySelectorAll('[role="tablist"]').forEach(tablist => {
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  function activate(tab) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        activate(tabs[next]);
        tabs[next].focus();
      }
    });
  });
  activate(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
});

const benchmarkBars = document.getElementById('benchmark-bars');
if (benchmarkBars) {
  const data = [
    { name: 'Default FL', time: 115.17, size: 0.810 },
    { name: 'NISP', time: 112.64, size: 0.317 },
    { name: 'SNIP', time: 73.04, size: 0.344 },
    { name: 'PUMA Default', time: 54.13, size: 0.144 },
    { name: 'PUMA 500mb', time: 54.92, size: 0.148 },
    { name: 'PUMA 100mb', time: 55.32, size: 0.153 }
  ];
  function renderMetric(metric) {
    const time = metric === 'time';
    const maximum = Math.max(...data.map(item => item[metric]));
    benchmarkBars.replaceChildren();
    for (const item of data) {
      const row = document.createElement('div');
      row.className = 'bar-row' + (item.name === 'PUMA Default' ? ' is-puma' : '');
      const name = document.createElement('span');
      name.textContent = item.name;
      const track = document.createElement('div');
      track.className = 'bar-track';
      track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = `${100 * item[metric] / maximum}%`;
      track.append(fill);
      const value = document.createElement('span');
      value.className = 'bar-value';
      value.textContent = item[metric].toFixed(time ? 2 : 3) + (time ? ' s' : ' MB');
      row.append(name, track, value);
      benchmarkBars.append(row);
    }
    document.getElementById('chart-title').textContent = time ? 'Average round time' : 'Average model per client';
    document.getElementById('chart-unit').textContent = time ? 'Seconds · lower is better' : 'Megabytes · lower is better';
    document.getElementById('insight-value').textContent = `${((1 - data[3][metric] / data[0][metric]) * 100).toFixed(1)}%`;
    document.getElementById('insight-title').textContent = time ? 'lower average round time' : 'smaller model per client';
    document.getElementById('insight-description').textContent = time ? '54.13 s compared with 115.17 s in the reported experiment.' : '0.144 MB compared with 0.810 MB in the reported experiment.';
    document.querySelectorAll('[data-metric]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === metric)));
  }
  document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => renderMetric(button.dataset.metric)));
  renderMetric('time');
}
