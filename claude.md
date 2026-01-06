# Claude Code Guidelines for webgpu-galaxy

## Blog Post Writing Style

When writing or updating BLOG.md, follow the **"Build, Break, Fix"** narrative pattern:

### The Pattern

1. **Build something** - Introduce a concept or implement a feature
2. **Show it breaking** - Demonstrate the problem/artifact/limitation that emerges
3. **Fix it** - Introduce the solution and explain why it works

### Why This Works

- Readers learn not just *what* to do, but *why* it's necessary
- Creates a natural narrative flow that keeps readers engaged
- Mimics the real development process, making it more authentic
- Each "fix" builds understanding of the underlying problem

### Example Structure

```markdown
### 3.2 The Naive Approach

Here's our first attempt at raymarching the disk...

[code]

This works, but if we look closely at the thin inner regions of the disk,
we see aliasing artifacts:

[image showing the problem]

The issue is that our fixed step size is larger than the disk thickness,
so rays can skip over the disk entirely.

### 3.3 Adaptive Stepping

The solution is to vary our step size based on proximity to thin geometry...

[improved code]

Now the rays take smaller steps when approaching the disk, catching those
thin features we were missing before.
```

### Specific Techniques to Demonstrate

1. **Aliasing from fixed step sizes** → **Adaptive stepping based on disk proximity**
2. **Banding artifacts** → **Step jitter to break up uniform sampling**
3. **Performance issues** → **Early termination and distance-based optimization**

### Tone Guidelines

- Conversational but technically precise
- Use "we" to include the reader in the journey
- Acknowledge mistakes/problems honestly - they're learning opportunities
- Connect each fix back to the underlying physics or math when relevant

## Code Organization

The simulation code in `blackhole.js` is organized pedagogically with section comments
matching the blog post structure. When modifying code, maintain this organization
and update corresponding blog sections.
