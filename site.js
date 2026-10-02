'use strict';
const dialog = document.querySelector('#figure-dialog');
const expanded = document.querySelector('#dialog-image');
const imageArea = document.querySelector('.dialog-image');
const zoomButton = document.querySelector('#zoom-toggle');
let lastFigure;
document.querySelectorAll('.figure-open').forEach(link => {
  link.addEventListener('click', event => {
    if (!dialog.showModal || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    lastFigure = link;
    expanded.src = link.href;
    expanded.alt = link.querySelector('img').alt;
    document.querySelector('#dialog-caption').textContent = link.dataset.caption;
    document.querySelector('#dialog-pdf').href = link.dataset.pdf;
    imageArea.classList.remove('full-resolution');
    zoomButton.setAttribute('aria-pressed', 'false');
    zoomButton.textContent = 'Full resolution';
    dialog.showModal();
    document.body.classList.add('dialog-open');
    document.querySelector('#dialog-close').focus();
  });
});
document.querySelector('#dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => {
  document.body.classList.remove('dialog-open');
  lastFigure?.focus({preventScroll:true});
});
zoomButton.addEventListener('click', () => {
  const full = imageArea.classList.toggle('full-resolution');
  zoomButton.setAttribute('aria-pressed', String(full));
  zoomButton.textContent = full ? 'Fit to window' : 'Full resolution';
});

function accessibleTabs(selector, callback) {
  const tabs = [...document.querySelectorAll(selector)];
  function activate(tab) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    callback(tab);
  }
  tabs.forEach((tab,index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next=(index+1)%tabs.length;
      if (event.key === 'ArrowLeft') next=(index-1+tabs.length)%tabs.length;
      if (event.key === 'Home') next=0;
      if (event.key === 'End') next=tabs.length-1;
      if (next !== undefined) { event.preventDefault(); tabs[next].focus(); activate(tabs[next]); }
    });
  });
  return activate;
}
accessibleTabs('[data-ablation]', tab => {
  document.querySelectorAll('[id^="ablation-"][role="tabpanel"]').forEach(panel => {
    panel.hidden = panel.id !== tab.getAttribute('aria-controls');
  });
});

fetch('assets/results.json').then(response => {
  if (!response.ok) throw new Error('Results unavailable');
  return response.json();
}).then(data => {
  function render(complexity) {
    const body = document.querySelector('#comparison-body');
    const maxima = data.metrics.map((_,index) => Math.max(...data.comparison.map(row => row[complexity][index])));
    body.replaceChildren();
    data.comparison.forEach(row => {
      const tr=document.createElement('tr');
      if (row.ours) tr.className='ours';
      const method=document.createElement('th'); method.scope='row'; method.textContent=row.method; tr.append(method);
      const backbone=document.createElement('td'); backbone.textContent=row.backbone; tr.append(backbone);
      row[complexity].forEach((value,index) => {
        const td=document.createElement('td');
        if (value === maxima[index]) { const strong=document.createElement('strong'); strong.textContent=value.toFixed(3); td.append(strong); }
        else td.textContent=value.toFixed(3);
        tr.append(td);
      });
      body.append(tr);
    });
    document.querySelector('#comparison-caption').textContent=`Baseline comparison for ${complexity} tasks`;
  }
  accessibleTabs('[data-complexity]', tab => {
    document.querySelector('#comparison-panel').setAttribute('aria-labelledby',tab.id);
    render(tab.dataset.complexity);
  });
  render('complex');
}).catch(() => {
  const tr=document.createElement('tr'); const td=document.createElement('td'); td.colSpan=7;
  const link=document.createElement('a');link.href='assets/results.json';link.textContent='View the numerical results data';td.append(link);tr.append(td);
  document.querySelector('#comparison-body').append(tr);
});
