# Stage 3.5: Results Analysis & Visualization

## Objective
Deep dive into the letter-publication matching results from Stage 3. Includes interactive visualizations, temporal analysis, correspondence network analysis, and detailed examination of high-confidence matches.

---

## 1. Setup & Load Results

### 1.1 Install Visualization Libraries

```python
!pip install -q matplotlib seaborn plotly pandas numpy scipy networkx

print("✓ Visualization libraries installed")
```

### 1.2 Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("✓ Libraries imported")
```

### 1.3 Mount Drive and Load Data

```python
from google.colab import drive
drive.mount('/content/drive')

DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"

# Load matches
matches_df = pd.read_csv(f"{DARWIN_BASE_PATH}/letter_publication_matches.csv")
high_conf_df = pd.read_csv(f"{DARWIN_BASE_PATH}/letter_publication_matches_high_confidence.csv")

# Load letter metadata
letter_metadata = pd.read_csv(f"{DARWIN_BASE_PATH}/darwin_letters_embeddings_metadata.csv")

print(f"✓ Loaded {len(matches_df):,} total matches")
print(f"✓ Loaded {len(high_conf_df):,} high-confidence matches")
print(f"✓ Loaded {len(letter_metadata):,} letters")
```

---

## 2. Overview Statistics

### 2.1 Basic Statistics

```python
print("="*70)
print("MATCHING RESULTS OVERVIEW")
print("="*70)

print("\n📊 CONFIDENCE DISTRIBUTION:")
print(matches_df['confidence'].value_counts())
print(f"\nHigh-confidence rate: {(matches_df['confidence']=='high').sum() / len(matches_df) * 100:.1f}%")

print("\n📅 TEMPORAL COVERAGE:")
print(f"  Letter years: {matches_df['letter_year'].min()}-{matches_df['letter_year'].max()}")
print(f"  Publication years: {matches_df['pub_year'].min()}-{matches_df['pub_year'].max()}")
print(f"  Average temporal gap: {matches_df['temporal_gap'].mean():.1f} years")
print(f"  Max temporal gap: {matches_df['temporal_gap'].max()} years")

print("\n🔢 SIMILARITY METRICS:")
print(f"  Mean similarity: {matches_df['similarity'].mean():.3f}")
print(f"  Median similarity: {matches_df['similarity'].median():.3f}")
print(f"  High-conf mean: {matches_df[matches_df['confidence']=='high']['similarity'].mean():.3f}")

print("\n📖 PUBLICATIONS:")
pub_summary = matches_df.groupby('publication').agg({
    'chunk_idx': 'nunique',
    'letter_idx': 'count',
    'similarity': 'mean'
}).round(3)
pub_summary.columns = ['unique_chunks', 'total_matches', 'avg_similarity']
print(pub_summary)

print("="*70)
```

---

## 3. Temporal Visualizations

### 3.1 Publication Timeline with Match Density

```python
fig = go.Figure()

# Add bars for each publication showing match counts
pub_stats = matches_df.groupby(['publication', 'pub_year']).size().reset_index(name='match_count')
pub_stats = pub_stats.sort_values('pub_year')

fig.add_trace(go.Bar(
    x=pub_stats['pub_year'],
    y=pub_stats['match_count'],
    text=pub_stats['publication'],
    textposition='auto',
    marker_color='steelblue',
    name='Total Matches'
))

# Add line for high-confidence matches
high_conf_stats = high_conf_df.groupby('pub_year').size().reset_index(name='high_conf_count')
fig.add_trace(go.Scatter(
    x=high_conf_stats['pub_year'],
    y=high_conf_stats['high_conf_count'],
    mode='lines+markers',
    name='High Confidence',
    line=dict(color='red', width=3),
    marker=dict(size=10)
))

fig.update_layout(
    title='Letter-Publication Matches Over Time',
    xaxis_title='Publication Year',
    yaxis_title='Number of Matches',
    height=500,
    hovermode='x unified'
)

fig.show()
```

### 3.2 Temporal Gap Distribution by Publication

```python
fig = px.box(
    matches_df,
    x='publication',
    y='temporal_gap',
    color='confidence',
    title='Temporal Gap Distribution by Publication',
    labels={'temporal_gap': 'Years Between Letter and Publication'},
    height=600
)

fig.update_xaxes(tickangle=45)
fig.show()

print("\n💡 INTERPRETATION:")
print("• Smaller temporal gaps = letters written closer to publication date")
print("• Large gaps might indicate early conceptual work")
print("• Negative gaps would indicate errors (letters after publication)")
```

### 3.3 Letter Activity Heatmap

```python
# Create heatmap: Publication vs Letter Decade
matches_df['letter_decade'] = (matches_df['letter_year'] // 10) * 10

heatmap_data = matches_df.groupby(['publication', 'letter_decade']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    cbar_kws={'label': 'Number of Matches'}
)
plt.title('Letter Activity Heatmap: Which Decades Influenced Which Publications?', fontsize=14, pad=20)
plt.xlabel('Letter Decade', fontsize=12)
plt.ylabel('Publication', fontsize=12)
plt.tight_layout()
plt.savefig(f"{DARWIN_BASE_PATH}/viz_letter_activity_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: viz_letter_activity_heatmap.png")
```

---

## 4. Similarity Analysis

### 4.1 Similarity Distribution by Confidence

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overall distribution
ax = axes[0, 0]
for conf in ['high', 'moderate', 'low']:
    data = matches_df[matches_df['confidence'] == conf]['similarity']
    ax.hist(data, alpha=0.6, label=conf, bins=30)
ax.set_title('Similarity Distribution by Confidence Level')
ax.set_xlabel('Similarity Score')
ax.set_ylabel('Frequency')
ax.legend()
ax.axvline(0.75, color='red', linestyle='--', label='High threshold')
ax.axvline(0.60, color='orange', linestyle='--', label='Moderate threshold')

# Violin plot by publication
ax = axes[0, 1]
pub_order = matches_df.groupby('publication')['similarity'].mean().sort_values(ascending=False).index
sns.violinplot(
    data=matches_df,
    y='publication',
    x='similarity',
    order=pub_order,
    ax=ax
)
ax.set_title('Similarity Distribution by Publication')
ax.set_xlabel('Similarity Score')

# CDF
ax = axes[1, 0]
sorted_sims = np.sort(matches_df['similarity'])
cdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
ax.plot(sorted_sims, cdf, linewidth=2)
ax.set_title('Cumulative Distribution of Similarity Scores')
ax.set_xlabel('Similarity Score')
ax.set_ylabel('Cumulative Probability')
ax.grid(True, alpha=0.3)

# Match count by publication
ax = axes[1, 1]
pub_counts = matches_df.groupby('publication').size().sort_values(ascending=False)
ax.barh(range(len(pub_counts)), pub_counts.values, alpha=0.7)
ax.set_yticks(range(len(pub_counts)))
ax.set_yticklabels(pub_counts.index, fontsize=8)
ax.set_xlabel('Number of Matches')
ax.set_title('Total Matches by Publication')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{DARWIN_BASE_PATH}/viz_similarity_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: viz_similarity_analysis.png")
```

### 4.2 Similarity vs Temporal Gap

```python
fig = px.scatter(
    matches_df.sample(10000),  # Sample for performance
    x='temporal_gap',
    y='similarity',
    color='confidence',
    facet_col='publication',
    facet_col_wrap=4,
    title='Similarity vs Temporal Gap (Sample of 10,000 matches)',
    labels={'temporal_gap': 'Years Between Letter & Publication', 'similarity': 'Similarity Score'},
    height=800,
    opacity=0.5
)

fig.update_xaxes(range=[0, 30])
fig.show()

print("\n💡 INTERPRETATION:")
print("• Look for patterns: Do older letters have lower similarity?")
print("• High similarity + small temporal gap = likely direct influence")
print("• High similarity + large temporal gap = long-term conceptual development")
```

---

## 5. Correspondence Network Analysis

### 5.1 Top Correspondents by Publication

```python
print("="*70)
print("TOP CORRESPONDENTS BY PUBLICATION")
print("="*70)

for pub in matches_df['publication'].unique():
    pub_matches = matches_df[matches_df['publication'] == pub]
    top_correspondents = pub_matches['letter_receiver'].value_counts().head(10)

    print(f"\n📖 {pub}:")
    for correspondent, count in top_correspondents.items():
        pct = (count / len(pub_matches)) * 100
        print(f"  {correspondent:30s}: {count:5d} matches ({pct:5.1f}%)")

print("="*70)
```

### 5.2 Correspondence Network Visualization (High Confidence)

```python
# Build network from high-confidence matches
G = nx.Graph()

# Count connections between publications and correspondents
for pub in high_conf_df['publication'].unique():
    pub_matches = high_conf_df[high_conf_df['publication'] == pub]
    top_5_correspondents = pub_matches['letter_receiver'].value_counts().head(5)

    for correspondent, count in top_5_correspondents.items():
        if count > 20:  # Only significant connections
            G.add_edge(pub, correspondent, weight=count)

# Draw network
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Node colors
pub_nodes = [n for n in G.nodes() if n in matches_df['publication'].unique()]
correspondent_nodes = [n for n in G.nodes() if n not in pub_nodes]

# Draw
nx.draw_networkx_nodes(G, pos, nodelist=pub_nodes, node_color='lightblue',
                       node_size=3000, alpha=0.7, label='Publications')
nx.draw_networkx_nodes(G, pos, nodelist=correspondent_nodes, node_color='lightcoral',
                       node_size=2000, alpha=0.7, label='Correspondents')

# Edge thickness by weight
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, width=[w/50 for w in weights], alpha=0.3)

nx.draw_networkx_labels(G, pos, font_size=8)

plt.title('Darwin Correspondence Network (High-Confidence Matches)', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{DARWIN_BASE_PATH}/viz_correspondence_network.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: viz_correspondence_network.png")
print("\n💡 INTERPRETATION:")
print("• Blue nodes = Publications")
print("• Red nodes = Correspondents")
print("• Edge thickness = Number of high-confidence matches")
```

---

## 6. Publication-Specific Deep Dives

### 6.1 Origin of Species Detailed Analysis

```python
print("="*70)
print("DEEP DIVE: ON THE ORIGIN OF SPECIES (1859)")
print("="*70)

origin_matches = matches_df[matches_df['publication'] == 'Origin'].copy()

print(f"\n📊 STATISTICS:")
print(f"  Total matches: {len(origin_matches):,}")
print(f"  High confidence: {(origin_matches['confidence']=='high').sum():,} ({(origin_matches['confidence']=='high').mean()*100:.1f}%)")
print(f"  Avg similarity: {origin_matches['similarity'].mean():.3f}")
print(f"  Avg temporal gap: {origin_matches['temporal_gap'].mean():.1f} years")
print(f"  Letters before pub: {(origin_matches['temporal_gap'] >= 0).sum():,} ({(origin_matches['temporal_gap'] >= 0).mean()*100:.1f}%)")

print(f"\n📅 TEMPORAL DISTRIBUTION:")
year_dist = origin_matches['letter_year'].value_counts().sort_index()
print(year_dist.tail(10))

print(f"\n👥 TOP CORRESPONDENTS:")
top_corr = origin_matches['letter_receiver'].value_counts().head(10)
for corr, count in top_corr.items():
    print(f"  {corr:30s}: {count:4d} matches")

# Visualize Origin matches over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Timeline of matches
year_counts = origin_matches['letter_year'].value_counts().sort_index()
ax1.bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7)
ax1.axvline(1859, color='red', linestyle='--', linewidth=2, label='Publication (1859)')
ax1.set_title('Origin of Species: Letter Matches by Year', fontsize=14)
ax1.set_xlabel('Letter Year')
ax1.set_ylabel('Number of Matches')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Similarity distribution by year
origin_high_conf = origin_matches[origin_matches['confidence'] == 'high']
year_sim = origin_high_conf.groupby('letter_year')['similarity'].agg(['mean', 'count']).reset_index()
year_sim = year_sim[year_sim['count'] > 10]  # Only years with >10 matches

ax2.scatter(year_sim['letter_year'], year_sim['mean'], s=year_sim['count']*2, alpha=0.6)
ax2.plot(year_sim['letter_year'], year_sim['mean'], color='steelblue', alpha=0.5)
ax2.axvline(1859, color='red', linestyle='--', linewidth=2, label='Publication (1859)')
ax2.set_title('Average Similarity of High-Confidence Matches by Year', fontsize=14)
ax2.set_xlabel('Letter Year')
ax2.set_ylabel('Average Similarity')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{DARWIN_BASE_PATH}/viz_origin_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Saved: viz_origin_analysis.png")
```

### 6.2 Top 20 High-Confidence Matches for Origin

```python
print("\n" + "="*70)
print("TOP 20 HIGH-CONFIDENCE MATCHES FOR ORIGIN OF SPECIES")
print("="*70)

origin_high = high_conf_df[high_conf_df['publication'] == 'Origin'].sort_values('similarity', ascending=False).head(20)

for idx, match in origin_high.iterrows():
    print(f"\n🔗 Match #{idx + 1} (Similarity: {match['similarity']:.3f})")
    print(f"   Letter: {match['letter_id']} ({match['letter_year']})")
    print(f"   To: {match['letter_receiver']}")
    print(f"   Temporal gap: {match['temporal_gap']} years")
    print(f"   Before pub: {'✓' if match['temporal_gap'] >= 0 else '✗'}")
    print(f"   Chunk preview: {match['chunk_text'][:150]}...")
    print("-" * 70)
```

---

## 7. Temporal Validation Analysis

### 7.1 Temporal Coherence by Publication

```python
fig = go.Figure()

# Calculate percentage of matches where letter comes before publication
temporal_df = matches_df.groupby(['publication', 'pub_year']).agg({
    'temporal_gap': lambda x: (x >= 0).mean(),
    'letter_idx': 'count'
}).reset_index()

temporal_df.columns = ['publication', 'pub_year', 'letters_before_pub_pct', 'total_matches']
temporal_df = temporal_df.sort_values('letters_before_pub_pct', ascending=False)

# Bar chart
fig.add_trace(go.Bar(
    x=temporal_df['publication'],
    y=temporal_df['letters_before_pub_pct'] * 100,
    text=[f"{pct:.1f}%" for pct in temporal_df['letters_before_pub_pct'] * 100],
    textposition='outside',
    marker_color=['green' if pct >= 0.80 else 'orange' if pct >= 0.60 else 'red'
                  for pct in temporal_df['letters_before_pub_pct']],
    name='Letters Before Publication'
))

# Add target line
fig.add_hline(y=80, line_dash="dash", line_color="green",
              annotation_text="80% Ideal", annotation_position="right")

fig.update_layout(
    title='Temporal Coherence: % of Matches Where Letter Predates Publication',
    xaxis_title='Publication',
    yaxis_title='Letters Before Publication (%)',
    height=500,
    showlegend=False
)

fig.update_xaxes(tickangle=45)
fig.show()

print("\n📊 TEMPORAL COHERENCE BREAKDOWN:")
for _, row in temporal_df.iterrows():
    status = "✅" if row['letters_before_pub_pct'] >= 0.80 else "🟡" if row['letters_before_pub_pct'] >= 0.60 else "🔴"
    before_count = int(row['letters_before_pub_pct'] * row['total_matches'])
    print(f"{status} {row['publication']:15s}: {row['letters_before_pub_pct']*100:5.1f}% ({before_count}/{int(row['total_matches'])} matches)")
```

### 7.2 Future Letters Analysis (Letters After Publication)

```python
print("\n" + "="*70)
print("FUTURE LETTERS ANALYSIS (Letters After Publication)")
print("="*70)

future_letters = matches_df[matches_df['temporal_gap'] < 0].copy()
print(f"\nTotal 'future' matches: {len(future_letters):,} ({len(future_letters)/len(matches_df)*100:.1f}%)")

print("\n📊 Future Letters by Publication:")
future_by_pub = future_letters.groupby('publication').agg({
    'letter_idx': 'count',
    'similarity': 'mean',
    'temporal_gap': 'mean'
}).round(3)
future_by_pub.columns = ['future_count', 'avg_similarity', 'avg_gap_years']
future_by_pub = future_by_pub.sort_values('future_count', ascending=False)
print(future_by_pub)

print("\n💡 POSSIBLE REASONS FOR POST-PUBLICATION MATCHES:")
print("  • Later editions/revisions of publications")
print("  • Darwin discussing published work with correspondents")
print("  • Continued research on same topics")
print("  • General thematic similarity")

# Show examples of future letters with high similarity
print("\n⚠️  High-Similarity Post-Publication Matches:")
high_sim_future = future_letters[future_letters['similarity'] > 0.75].sort_values('similarity', ascending=False).head(10)

for idx, match in high_sim_future.iterrows():
    print(f"\n  Similarity: {match['similarity']:.3f} | {match['publication']} ({match['pub_year']})")
    print(f"  Letter: {match['letter_id']} ({match['letter_year']}) → {match['letter_receiver']}")
    print(f"  Gap: {match['temporal_gap']} years")
```

---

## 8. Interactive Exploration Dashboard

### 8.1 Filterable Match Explorer

```python
# Create interactive table with filters
from IPython.display import HTML

# Prepare data for display
display_df = matches_df[['publication', 'pub_year', 'letter_id', 'letter_year',
                         'letter_receiver', 'similarity', 'confidence',
                         'temporal_gap', 'chunk_text']].copy()

display_df['chunk_text'] = display_df['chunk_text'].str[:100] + '...'
display_df = display_df.sort_values('similarity', ascending=False)

print("="*70)
print("INTERACTIVE MATCH EXPLORER")
print("="*70)
print("\nUse the filters below to explore matches:")
print("• Filter by publication, confidence, temporal accuracy, etc.")
print("• Sort by any column")
print("\n")

# Display top 100 matches
print(display_df.head(100).to_string(index=False))

# Save full dataset for external exploration
export_path = f"{DARWIN_BASE_PATH}/matches_for_exploration.csv"
display_df.to_csv(export_path, index=False)
print(f"\n✓ Full dataset exported: {export_path}")
print("  (Open in Excel/Google Sheets for interactive filtering)")
```

### 8.2 Publication Comparison Matrix

```python
# Create similarity matrix between publications based on shared correspondents
pub_list = matches_df['publication'].unique()
similarity_matrix = np.zeros((len(pub_list), len(pub_list)))

for i, pub1 in enumerate(pub_list):
    for j, pub2 in enumerate(pub_list):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            # Count shared correspondents (weighted by match count)
            pub1_corr = set(matches_df[matches_df['publication'] == pub1]['letter_receiver'])
            pub2_corr = set(matches_df[matches_df['publication'] == pub2]['letter_receiver'])
            shared = len(pub1_corr & pub2_corr)
            total = len(pub1_corr | pub2_corr)
            similarity_matrix[i, j] = shared / total if total > 0 else 0

plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    xticklabels=pub_list,
    yticklabels=pub_list,
    cbar_kws={'label': 'Correspondence Overlap'}
)
plt.title('Publication Similarity Based on Shared Correspondents', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f"{DARWIN_BASE_PATH}/viz_publication_similarity_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: viz_publication_similarity_matrix.png")
print("\n💡 INTERPRETATION:")
print("• High values = publications share many of the same correspondents")
print("• May indicate topical similarity or chronological proximity")
```

---

## 9. Export Summary Report

### 9.1 Generate Summary Statistics

```python
summary_stats = {
    "analysis_date": pd.Timestamp.now().isoformat(),
    "total_matches": len(matches_df),
    "high_confidence_matches": len(high_conf_df),
    "publications_analyzed": len(matches_df['publication'].unique()),
    "unique_letters_matched": matches_df['letter_idx'].nunique(),
    "letters_before_pub_pct": (matches_df['temporal_gap'] >= 0).mean(),
    "average_similarity": matches_df['similarity'].mean(),
    "average_temporal_gap": matches_df['temporal_gap'].mean(),
    "confidence_distribution": matches_df['confidence'].value_counts().to_dict(),
    "best_publications": temporal_df.head(5)[['publication', 'letters_before_pub_pct']].to_dict('records'),
    "needs_improvement": temporal_df.tail(3)[['publication', 'letters_before_pub_pct']].to_dict('records')
}

# Save report
import json
report_path = f"{DARWIN_BASE_PATH}/stage3_analysis_report.json"
with open(report_path, 'w') as f:
    json.dump(summary_stats, f, indent=2, default=str)

print("✓ Summary report saved:", report_path)
```

### 9.2 Print Executive Summary

```python
print("\n" + "="*70)
print("EXECUTIVE SUMMARY: LETTER-PUBLICATION MATCHING RESULTS")
print("="*70)

print(f"\n📊 OVERALL METRICS:")
print(f"  Total matches: {summary_stats['total_matches']:,}")
print(f"  High-confidence: {summary_stats['high_confidence_matches']:,} ({summary_stats['high_confidence_matches']/summary_stats['total_matches']*100:.1f}%)")
print(f"  Letters before pub: {summary_stats['letters_before_pub_pct']:.1%}")
print(f"  Average similarity: {summary_stats['average_similarity']:.3f}")

print(f"\n✅ BEST TEMPORAL COHERENCE:")
for pub in summary_stats['best_publications']:
    print(f"  • {pub['publication']:15s}: {pub['letters_before_pub_pct']*100:.1f}% before pub")

print(f"\n⚠️  PUBLICATIONS NEEDING REVIEW:")
for pub in summary_stats['needs_improvement']:
    print(f"  • {pub['publication']:15s}: {pub['letters_before_pub_pct']*100:.1f}% before pub")

print(f"\n🎯 CONCLUSION:")
if summary_stats['letters_before_pub_pct'] >= 0.80:
    print("  ✅ EXCELLENT: Strong temporal coherence - most letters predate publications")
elif summary_stats['letters_before_pub_pct'] >= 0.70:
    print("  ✅ GOOD: Acceptable temporal coherence")
elif summary_stats['letters_before_pub_pct'] >= 0.60:
    print("  🟡 ACCEPTABLE: Some post-publication matches may indicate later editions/discussions")
else:
    print("  🔴 REVIEW: High rate of post-publication matches - review matching strategy")

print("\n📁 VISUALIZATIONS GENERATED:")
viz_files = [
    "viz_letter_activity_heatmap.png",
    "viz_similarity_analysis.png",
    "viz_correspondence_network.png",
    "viz_origin_analysis.png",
    "viz_publication_similarity_matrix.png"
]
for vf in viz_files:
    print(f"  • {vf}")

print("\n" + "="*70)
```

---

## 10. Next Steps & Recommendations

```python
print("\n" + "="*70)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("="*70)

# Identify publications with low temporal coherence
low_coherence_pubs = temporal_df[temporal_df['letters_before_pub_pct'] < 0.70]['publication'].tolist()

if low_coherence_pubs:
    print(f"\n📋 PUBLICATIONS TO REVIEW ({len(low_coherence_pubs)}):")
    for pub in low_coherence_pubs:
        pub_data = temporal_df[temporal_df['publication'] == pub].iloc[0]
        print(f"\n  {pub} ({pub_data['letters_before_pub_pct']*100:.1f}% letters before pub):")

        # Analyze why coherence is low
        pub_matches = matches_df[matches_df['publication'] == pub]
        future_rate = (pub_matches['temporal_gap'] < 0).mean()
        avg_gap = pub_matches['temporal_gap'].mean()

        print(f"    • Post-publication rate: {future_rate*100:.1f}%")
        print(f"    • Average temporal gap: {avg_gap:.1f} years")

        if avg_gap < -5:
            print(f"    → Consider: May match later editions or post-publication discussions")
        if future_rate > 0.5:
            print(f"    → Consider: Semantic overlap with general scientific language")

print("\n💡 GENERAL RECOMMENDATIONS:")
print("  1. Investigate post-publication matches - may reveal later editions/discussions")
print("  2. Consider adding correspondent weighting (Hooker, Wallace, etc.)")
print("  3. Test different similarity thresholds for edge cases")
print("  4. Validate high-confidence matches with historical records")
print("  5. Cross-reference with Darwin Correspondence Project metadata")

print("\n🎓 RESEARCH APPLICATIONS:")
print("  • Identify which letters influenced specific publication passages")
print("  • Track conceptual development across decades")
print("  • Map Darwin's intellectual network")
print("  • Discover understudied correspondences")
print("  • Validate historical claims about publication sources")

print("="*70)
```

---

## Notes

- **Processing time**: ~5-10 minutes (mostly visualization)
- **Memory usage**: ~2-4 GB (for full matches dataset)
- **Visualizations**: 5 PNG files + interactive plots
- **Export**: CSV for Excel/Sheets exploration, JSON summary report

---

## Interpretation Guide

### Similarity Scores:
- **≥0.75 (high)**: Strong semantic connection, likely direct influence
- **0.60-0.75 (moderate)**: Related concepts, possible influence
- **0.45-0.60 (low)**: Weak connection, general topic overlap

### Temporal Accuracy:
- **≥70%**: Excellent - matches align with known work periods
- **60-70%**: Good - most matches are chronologically valid
- **<60%**: Review needed - many anachronistic matches

### Temporal Gap:
- **0-5 years**: Recent correspondence, likely direct influence
- **5-15 years**: Medium-term development, conceptual evolution
- **>15 years**: Long-term ideas, early conceptual work
