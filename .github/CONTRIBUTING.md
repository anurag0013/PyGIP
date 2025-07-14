# 🧩 Contributing to This Project

Thank you for your interest in contributing! We welcome all contributions — bug reports, feature requests, pull
requests, and documentation improvements.

Please follow the guidelines below to ensure a smooth contribution process.

---

## 🐛 Reporting Bugs

1. Check if the bug has already been reported.
2. Open a new [bug report](https://github.com/your-org/your-repo/issues/new?template=bug_report.md).
3. Include:
    - Clear reproduction steps
    - Expected vs. actual behavior
    - Logs, screenshots, or minimal code snippets if possible

---

## 🚀 Suggesting Features

1. Check if a similar feature request already exists.
2. Open a new [feature request](https://github.com/your-org/your-repo/issues/new?template=feature_request.md).
3. Include:
    - Motivation and use case
    - Desired functionality
    - Alternatives you considered

---

## 🔧 Submitting Pull Requests

> All code changes should be proposed via Pull Request (PR) from a feature branch — **do not commit directly to `main`
**.

### Steps:

1. Fork the repository
2. Create a new branch:

```shell
git checkout -b feat/your-feature-name
```

3. Commit your changes:

```shell
git push origin feat/your-feature-name
```

4. Push to your fork:

```shell
git push origin feat/your-feature-name
```

5. Open a PR against the `main` branch

---

## 🔧 Internal Code Submission Guideline

For internal team members with write access to the repository:

1. Always Use Feature/Fix Branches

- Never commit directly to the main or develop branch.
- Create a new branch for each feature, bug fix.

```shell
git checkout -b feat/your-feature-name
```

```shell
git checkout -b fix/your-fix-name
```

2. Keep Commits Clean & Meaningful

- feat: add data loader for graph dataset
- fix: resolve crash on edge cases

Use clear commit messages following the format:

```shell
<type>: <summary>
```

3. Test Before Pushing

- Test your implementation in `example.py`, and compare the performance with the results in original paper.

4. Push to Internal Branch

- Always run `git pull origin pygip-release` before pushing your changes
- Submit a pull request targeting the `pygip-release` branch
- Write a brief summary describing the features you’ve added, how to run your method, and how to evaluate its
  performance

Push to the remote feature branch.

```shell
git push origin feat/your-feature-name
```

---

## 📄 Code Style & Testing

- Follow existing code conventions
- Use meaningful names and comments
- Add tests for new features or bug fixes
- Run all tests before submitting a PR

---

## 💬 Questions or Help?

- Use Discussions for general questions
- Feel free to open an issue if something is unclear

---

Thank you for contributing! 🙌