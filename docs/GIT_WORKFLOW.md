# Git Workflow Guide - Credit Risk ML Pipeline

## Branching Strategy

This project uses an adapted **Git Flow** branching strategy:

### Main Branches

- **`main`** (or `master`): Production code, always stable
- **`develop`**: Development branch, feature integration

### Feature Branches

- **`feature/step-11-production-deployment`**: Step 11 implementation
- **`feature/step-12-monitoring`**: Monitoring implementation (future)
- **`feature/bugfix-*`**: Bug fixes
- **`feature/refactor-*`**: Refactoring

## Useful Commands

### Check Status
```bash
git status
git remote -v  # View configured remote repositories
```

### Create and Work on Branches
```bash
# Create branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/step-11-production-deployment

# Make commits
git add .
git commit -m "feat: implement model serving API"

# Push branch
git push -u origin feature/step-11-production-deployment
```

### Merge Features
```bash
# After review, merge into develop
git checkout develop
git merge feature/step-11-production-deployment
git push origin develop

# When ready for production
git checkout main
git merge develop
git push origin main
```

## Commit Convention

We use **Conventional Commits**:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Refactoring
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance

Example: `feat: add model serving API with FastAPI`
