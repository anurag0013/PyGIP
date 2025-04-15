## External Pull Request Guideline

External contributors are welcome! Please follow the steps below to submit a clean and reviewable pull request:

### 🔧 Step 1: Fork the Repository

- Navigate to the repository page.
- Click on the **Fork** button in the upper-right corner to create your own copy.

### 💻 Step 2: Clone Your Fork

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 🌱 Step 3: Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

🔒 Do not make changes directly on your main branch.

### 🛠 Step 4: Make Your Changes

- Follow the project’s code style and structure.
- Add comments and documentation where appropriate.
- Write or update tests to cover your changes.

### ✅ Step 5: Run Tests

Provide examples about how to run, and test performance.

### 📤 Step 6: Commit and Push

```shell
git add .
git commit -m "feat: concise summary of your change"
git push origin feature/your-feature-name
```

### Step 7: Open a Pull Request

- Go to your fork on GitHub.
- Click “Compare & pull request”
- Complete the PR template:
    - What the PR does
    - Why it’s needed
    - How to test
- Link any related issue (e.g., Closes #42)

🎯 Make sure your PR targets the correct base branch.

### 🔁 Step 8: Respond to Review Feedback

- Push follow-up commits to the same branch.
- Mark conversations as resolved.
- Actively engage with reviewers.

