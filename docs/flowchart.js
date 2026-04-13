/**
 * Mermaid flowchart for the recommender data flow.
 * Usage: render this string with Mermaid in a browser or CLI tool.
 */

const mermaidDiagram = `flowchart LR
  Input[User Preferences]
  Process[The Loop: judge each song in data/songs.csv using scoring logic]
  Output[Top-K Recommendations]
  Input --> Process --> Output
`;

if (require.main === module) {
  // Print the diagram when run directly: `node docs/flowchart.js`
  console.log(mermaidDiagram);
}

module.exports = mermaidDiagram;
