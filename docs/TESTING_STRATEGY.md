# Testing Strategy - Ensuring No Tests Get Skipped

## Problem

Chaos and E2E tests are excluded from CI to keep it fast (~3-4 min), but we need to ensure they still run regularly.

## Solution: Multi-Tier Testing Strategy

### Tier 1: Fast CI (Every Push/PR) ⚡

**Runs:** Unit + Integration tests
**Time:** ~3-4 minutes
**Coverage:** 82%+
**Purpose:** Fast feedback, blocks broken code

```yaml
# .github/workflows/ci.yml
pytest tests/unit/ tests/integration/ --cov=src --cov=app --cov-fail-under=75
```

### Tier 2: Nightly Builds 🌙

**Runs:** Unit + Integration + Chaos tests
**Time:** ~8-10 minutes
**Schedule:** Every night at 2 AM UTC
**Purpose:** Catch issues that slip through fast CI

```bash
# Automatically runs via .github/workflows/nightly.yml
# Or trigger manually: Actions → Nightly Full Test Suite → Run workflow
```

### Tier 3: Pre-Release Validation 🚀

**Runs:** All tests + Security scans + Evaluation
**Time:** ~10-15 minutes
**Trigger:** When creating version tags (v1.0.0, v2.0.0, etc.)
**Purpose:** Ensure release quality

```bash
# Automatically runs when you create a release tag
git tag v1.0.0
git push origin v1.0.0
```

### Tier 4: Manual Testing (Before Major Changes) 🔧

**Runs:** All tests including E2E and Load tests
**Time:** ~15-20 minutes
**When:** Before merging major features or releases

```bash
# Run locally before major releases
./scripts/run_tests.sh           # Unit + Integration + Chaos
./scripts/run_load_tests.sh      # Load/performance tests
./scripts/run_e2e_tests.sh       # End-to-end UI tests
```

---

## Enforcement Mechanisms

### 1. **Automated Nightly Builds**

- ✅ Runs automatically every night
- ✅ Catches regressions early
- ✅ No manual intervention needed
- ⚠️ Failures create GitHub notifications

### 2. **Pre-Release Workflow**

- ✅ Runs automatically on version tags
- ✅ Blocks release if tests fail
- ✅ Comprehensive validation

### 3. **Pre-Push Hook (Optional)**

Run all tests before pushing:

```bash
# Add to .git/hooks/pre-push
#!/bin/bash
echo "🧪 Running full test suite..."
./scripts/run_tests.sh
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix before pushing."
    exit 1
fi
echo "✅ All tests passed!"
```

### 4. **Pull Request Template**

Add checklist to `.github/pull_request_template.md`:

```markdown
## Pre-Merge Checklist

- [ ] All CI tests pass
- [ ] Ran `./scripts/run_tests.sh` locally (includes chaos tests)
- [ ] Updated tests for new features
- [ ] Documentation updated
```

### 5. **Branch Protection Rules**

In GitHub Settings → Branches → main:

- ✅ Require status checks to pass before merging
- ✅ Require "Unit & Integration Tests" to pass
- ✅ Require "Nightly Full Test Suite" to pass (if recent)

---

## Monitoring Test Coverage

### GitHub Actions Dashboard

1. Go to: **Actions** tab
2. Check: **Nightly Full Test Suite** status
3. Review: Failed runs immediately

### Local Monitoring

```bash
# Check what tests exist
find tests/ -name "test_*.py" -type f

# Count tests
pytest --collect-only tests/ | grep "test session starts"

# Run specific test types
pytest tests/unit/ -v           # Fast
pytest tests/integration/ -v    # Medium
pytest tests/chaos/ -v          # Slow
pytest tests/e2e/ -v            # Very slow
```

---

## Best Practices

### ✅ DO:

- Run `./scripts/run_tests.sh` before major commits
- Check nightly build status regularly
- Run full suite before creating releases
- Add tests for new features
- Keep CI fast (< 5 minutes)

### ❌ DON'T:

- Skip tests because "it's just a small change"
- Ignore nightly build failures
- Merge PRs with failing tests
- Remove tests to make CI faster
- Assume CI covers everything

---

## Quick Reference

| Test Type       | CI  | Nightly | Release | Manual |
| --------------- | :-: | :-----: | :-----: | :----: |
| **Unit**        | ✅  |   ✅    |   ✅    |   ✅   |
| **Integration** | ✅  |   ✅    |   ✅    |   ✅   |
| **Chaos**       | ❌  |   ✅    |   ✅    |   ✅   |
| **E2E**         | ❌  |   ❌    |   ⚠️    |   ✅   |
| **Load**        | ❌  |   ❌    |   ⚠️    |   ✅   |

**Legend:**

- ✅ Always runs
- ⚠️ Optional/conditional
- ❌ Excluded

---

## Troubleshooting

### "I forgot to run chaos tests before merging"

→ Nightly build will catch it and notify you

### "Nightly build is failing"

→ Fix immediately or revert the problematic commit

### "How do I know if all tests ran?"

→ Check GitHub Actions for green checkmarks on all workflows

### "CI is too slow again"

→ Review test execution times: `pytest --durations=10`

---

## Summary

**Problem:** Chaos/E2E tests excluded from CI might get skipped
**Solution:** Multi-tier testing with automated nightly builds and pre-release validation

This ensures:

- ✅ Fast CI for quick feedback
- ✅ Comprehensive testing via nightly builds
- ✅ Release quality via pre-release workflow
- ✅ No tests get permanently skipped
