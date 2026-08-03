# Project Rules Index

This directory contains modular, focused rules for the Edge-Cloud Code Analysis project. Each file covers a specific aspect of the system.

## 📚 Rule Files Overview

### Core Rules

**[proj.md](proj.md)** - Main project rules (400 lines)
- Quick overview and architecture
- Data flow patterns
- Key implementation rules
- Configuration hierarchy
- Common patterns
- **Start here** for project understanding

### Technical Documentation

**[architecture.md](architecture.md)** - System design (500 lines)
- Speculative decoding pattern
- Component architecture (Edge/Cloud/Core)
- Data models and flow
- Async architecture
- Extension points
- Performance optimization

**[modules.md](modules.md)** - Module reference (600 lines)
- Detailed API documentation for each module
- Function signatures and usage
- Key classes and methods
- Line number references
- Code examples for each component

### Development Guidelines

**[coding-standards.md](coding-standards.md)** - Code style (500 lines)
- Python style guide (PEP 8 + project conventions)
- Async/await patterns
- Pydantic model standards
- Error handling
- Logging standards
- Configuration management
- Common mistakes to avoid

**[testing.md](testing.md)** - Testing guide (400 lines)
- Testing philosophy
- Unit test patterns
- Async testing
- Mocking external services
- Fixtures and parametrization
- Coverage requirements
- CI/CD integration

### Operational Guides

**[usage-guide.md](usage-guide.md)** - How to use (500 lines)
- Quick start commands
- Common workflows (CI/CD, incremental)
- Configuration scenarios (cost/quality/speed)
- Tuning parameters
- Output formats
- Troubleshooting
- Advanced usage

**[deployment.md](deployment.md)** - Operations (500 lines)
- Local development setup
- Configuration management
- Production deployment (single-server, Docker, K8s)
- Database management
- Monitoring and metrics
- Backup & recovery
- Security best practices

### Reference

**[project-summary.md](project-summary.md)** - Status & stats (400 lines)
- Project information and statistics
- Completed features
- Architecture highlights
- Performance characteristics
- Configuration quick reference
- Known limitations
- Future roadmap

## 📖 Reading Guide

### For New Contributors
1. Start: **proj.md** - Get overview
2. Then: **architecture.md** - Understand design
3. Next: **coding-standards.md** - Learn conventions
4. Before coding: **modules.md** - Find what you need

### For Users
1. Start: **usage-guide.md** - Learn commands
2. Then: **deployment.md** - Set up environment
3. Reference: **proj.md** - Configuration options

### For Maintainers
1. Review: **project-summary.md** - Current status
2. Operations: **deployment.md** - Deployment patterns
3. Quality: **testing.md** - Test requirements
4. Standards: **coding-standards.md** - Code review checklist

## 📏 File Size Summary

| File | Lines | Focus | Priority |
|------|-------|-------|----------|
| proj.md | ~400 | Quick reference | ⭐⭐⭐ Essential |
| architecture.md | ~500 | System design | ⭐⭐⭐ Essential |
| modules.md | ~600 | API reference | ⭐⭐ Important |
| coding-standards.md | ~500 | Code style | ⭐⭐⭐ Essential |
| testing.md | ~400 | Testing | ⭐⭐ Important |
| usage-guide.md | ~500 | How-to | ⭐⭐⭐ Essential |
| deployment.md | ~500 | Operations | ⭐⭐ Important |
| project-summary.md | ~400 | Status | ⭐ Reference |
| **Total** | **~3,800** | **Complete** | |

All files kept under 600 lines for easy navigation.

## 🎯 Quick Lookup

### "How do I...?"

**Analyze code**
→ usage-guide.md § Basic Commands

**Add new language**
→ architecture.md § Extension Points

**Write a test**
→ testing.md § Writing Unit Tests

**Deploy to production**
→ deployment.md § Production Deployment

**Understand confidence scoring**
→ modules.md § edge/confidence_scorer.py

**Adjust upload threshold**
→ proj.md § Configuration Hierarchy

**Add logging**
→ coding-standards.md § Logging Standards

**Mock Ollama in tests**
→ testing.md § Mocking External Services

### "What is...?"

**Speculative decoding**
→ architecture.md § Design Pattern

**Strategy manager**
→ modules.md § core/strategy_manager.py

**Hotspot detection**
→ modules.md § edge/ast_analyzer.py

**Budget controller**
→ modules.md § core/budget_controller.py

**Data flow**
→ proj.md § Data Flow

### "Where is...?"

**Upload decision logic**
→ core/strategy_manager.py:should_upload()

**Confidence calculation**
→ edge/confidence_scorer.py:calibrate_confidence()

**Cloud API call**
→ cloud/client.py:verify()

**AST parsing**
→ edge/ast_analyzer.py:analyze_code()

**Budget tracking**
→ core/budget_controller.py:record_expense()

## 🔄 Update Guidelines

When updating rules:

1. **Keep files focused** - One topic per file
2. **Limit file size** - Max 600 lines
3. **Provide examples** - Show, don't just tell
4. **Reference code** - Link to actual implementation
5. **Update index** - Keep this README in sync

## 📝 Rule Writing Standards

Good rules are:
- ✅ **Specific**: "Use `@retry` decorator" not "Handle errors"
- ✅ **Actionable**: Provide code examples
- ✅ **Referenced**: Link to actual files/lines
- ✅ **Current**: Match actual implementation
- ✅ **Concise**: Under 600 lines per file

Bad rules:
- ❌ Vague: "Write good code"
- ❌ Outdated: Doesn't match implementation
- ❌ Too long: Over 600 lines
- ❌ No examples: Just theory

## 🔍 Search Tips

### In Cursor
- `Cmd/Ctrl+P` → Type `@.cursor/rules/` to see all rules
- Open specific file: `@.cursor/rules/modules.md`
- Search within: Open file and `Cmd/Ctrl+F`

### From Command Line
```bash
# Search all rules
grep -r "keyword" .cursor/rules/

# Find specific pattern
grep -l "StrategyManager" .cursor/rules/*.md

# Count lines
wc -l .cursor/rules/*.md
```

## 📦 Related Files

Outside `.cursor/rules/`:
- `README.md` - User-facing documentation
- `requirements.txt` - Dependencies
- `config/settings.yaml` - Configuration
- `examples/sample_code/` - Test files
- `tests/` - Unit tests

## 🤝 Contributing to Rules

When you modify code, also update:
1. Relevant rule file(s)
2. Code examples if API changed
3. Line number references if needed
4. This index if adding new file

---

**Quick Links**:
- Main Project Rules: [proj.md](proj.md)
- Architecture Details: [architecture.md](architecture.md)
- Module Reference: [modules.md](modules.md)
- User Guide: [usage-guide.md](usage-guide.md)

**Version**: v0.1.0 - Complete modular rules

