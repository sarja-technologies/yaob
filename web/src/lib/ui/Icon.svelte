<script lang="ts">
  // The few icons the app needs, drawn on a 16px grid with a 1.5px stroke.
  let { name, size = 16 }: { name: string; size?: number } = $props();
  const paths: Record<string, string> = {
    pan: 'M8 1.5v13M1.5 8h13M8 1.5 6 3.5M8 1.5l2 2M8 14.5l-2-2M8 14.5l2-2M1.5 8l2-2M1.5 8l2 2M14.5 8l-2-2M14.5 8l-2 2',
    select: 'M2.5 2.5h3M8 2.5h3M13.5 2.5v3M13.5 8v3M13.5 13.5h-3M8 13.5H5M2.5 13.5v-3M2.5 8V5',
    connect: 'M5.6 10.4l4.8-4.8M4 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM12 6a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z',
    note: 'M2.5 2.5h11v7l-4 4h-7zM13.5 9.5h-4v4',
    group: 'M2.5 4.5h11v9h-11zM2.5 4.5v-2h5v2',
    back: 'M9.5 3.5 5 8l4.5 4.5',
    forward: 'M6.5 3.5 11 8l-4.5 4.5',
    plus: 'M8 3v10M3 8h10',
    minus: 'M3 8h10',
    fit: 'M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10',
    fitsel: 'M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10M6 6h4v4H6z',
    views: 'M4 2.5h8v11l-4-3-4 3z',
    undo: 'M5.5 3 2.5 6l3 3M2.5 6h7a4 4 0 0 1 0 8h-3',
    redo: 'M10.5 3l3 3-3 3M13.5 6h-7a4 4 0 0 0 0 8h3',
    close: 'M3.5 3.5l9 9M12.5 3.5l-9 9',
    detach: 'M6 2.5H2.5v11h11V10M9 2.5h4.5V7M13.5 2.5 7.5 8.5',
    dock: 'M2.5 2.5h11v11h-11zM10 2.5v11',
    dockleft: 'M2.5 2.5h11v11h-11zM6 2.5v11',
    collapseR: 'M6 3.5 10.5 8 6 12.5M13 2.5v11',
    collapseL: 'M10 3.5 5.5 8l4.5 4.5M3 2.5v11',
    chevron: 'M4 6l4 4 4-4',
    chevronR: 'M6 4l4 4-4 4',
    more: 'M3.5 8h.01M8 8h.01M12.5 8h.01',
    search: 'M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10ZM10.7 10.7l3.3 3.3',
    panels: 'M2.5 2.5h11v11h-11zM6 2.5v11M10.5 2.5v11',
    arrowR: 'M3 8h10M9.5 4.5 13 8l-3.5 3.5',
    arrowL: 'M13 8H3M6.5 4.5 3 8l3.5 3.5',
    stop: 'M4 4h8v8H4z',
    send: 'M2.5 8h10M9 4.5 12.5 8 9 11.5',
  };
</script>

<svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width={name === 'more' ? 2.2 : 1.5} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
  ><path d={paths[name] ?? ''} /></svg
>
