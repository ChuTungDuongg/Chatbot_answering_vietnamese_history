// Display-only interpretation of explicit source metadata; years are never sources.
export function indexedSources(sources = []) {
  return new Map(sources.map((source, index) => [source.display_index ?? index + 1, source]));
}

export function displayAnswer(content = "", sources = []) {
  const ids = new Map(sources.map((source, index) => [String(source.source_id ?? source.chunk_id ?? source.id ?? ""), source.display_index ?? index + 1]));
  return content.replace(/\[([^\]\n[]+)\]/g, (bracket, id) => {
    if (/^\d+$/.test(id) || !ids.has(id)) return bracket;
    return `[${ids.get(id)}]`;
  });
}

export function remarkSourceCitations({ sources = [] } = {}) {
  const known = indexedSources(sources);
  return (tree) => {
    function visit(node) {
      if (!node.children || ["link", "code", "inlineCode", "html", "image"].includes(node.type)) return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text") { visit(child); return [child]; }
        const parts = [];
        let position = 0;
        for (const match of child.value.matchAll(/\[([1-9]\d?)\]/g)) {
          const index = Number(match[1]);
          // Only request-local display indices supplied by the API become controls.
          if (!known.has(index)) continue;
          parts.push({ type: "text", value: child.value.slice(position, match.index) });
          parts.push({ type: "link", url: `#source-${index}`, children: [{ type: "text", value: match[0] }] });
          position = match.index + match[0].length;
        }
        parts.push({ type: "text", value: child.value.slice(position) });
        return parts;
      });
    }
    visit(tree);
  };
}
