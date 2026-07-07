# Contribution Guide

- [Getting Started](#getting-started)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Development Guide](#development-guide)
  - [Code Style](#code-style)
  - [Fork-Pull Development Mode](#fork-pull-development-mode)
  - [Code Gate Exception Handling](#code-gate-exception-handling)
  - [ISSUE Specification](#issue-specification)
  - [Submitting a PR](#submitting-a-pr)

<h2 id="getting-started">Getting Started</h2>

- Fork the Triton-Ascend repository on [GitHub](https://github.com/triton-lang/triton-ascend).
- Read the [README.md](https://github.com/triton-lang/triton-ascend/blob/main/README.md) for project information and building the development environment.

<h2 id="developer-certificate-of-origin-dco">Developer Certificate of Origin (DCO)</h2>

All commits must include a `Signed-off-by:` line. Use `git commit -s` to add it automatically:

```bash
git commit -s -m "your commit message"
```

This automatically appends a line `Signed-off-by: Your Name <your.email@example.com>` at the end of the commit message, confirming your acknowledgment of the contribution's origin and authorization.

<h2 id="development-guide">Development Guide</h2>

- **[Code Style](#code-style)**
- **[Fork-Pull Development Mode](#fork-pull-development-mode)**
- **[Code Gate Exception Handling](#code-gate-exception-handling)**
- **[ISSUE Specification](#issue-specification)**
- **[Submitting a PR](#submitting-a-pr)**

<h2 id="code-style">Code Style</h2>

Please follow the coding styles below to make Triton Ascend easy to develop, maintain, and review.

- Coding Guidelines

  Use the unified coding style of the Triton Ascend community. The recommended Python coding style is [PEP 8 Coding Style](https://pep8.org/), and the recommended C++ coding style is [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy), [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [ShellCheck](https://github.com/koalaman/shellcheck), and [pylint](https://pylint.org/) to check code formatting. It is recommended to install these plugins in your IDE.

- Unit Testing Guidelines

  Use the unified unit testing style of the Triton Ascend community. The recommended Python unit testing style is [pytest](http://www.pytest.org/en/latest/), and the recommended C++ unit testing style is [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md). The design intent of a test case should be reflected by its comment name. For test case design, refer to the [gather test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/custom_op/test_gather_load.py) and the [layer_norm test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/05-layer-norm.py).

- Refactoring Guidelines

  We encourage developers to refactor our code to eliminate [code smells](https://en.wikipedia.org/wiki/Code_smell). Refactored code should also follow the coding style and testing style requirements. When you receive warnings, you need to refactor the code to be merged.

<h2 id="fork-pull-development-mode">Fork-Pull Development Mode</h2>

1. Fork the Triton Ascend Project

   Before submitting your code to the Triton Ascend project, ensure you have forked the Triton Ascend project to your own repository. You will subsequently develop on your forked project and merge changes into the Triton Ascend project via Pull Requests. This means there is parallel development between the Triton Ascend repository and your own repository, so please maintain consistency between them.

2. Clone the Remote Repository

   Use git to clone your forked Triton Ascend project and add the upstream repository:

   ```shell
   git clone https://github.com/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
   git remote add upstream https://github.com/triton-lang/triton-ascend.git
   ```

3. Develop Code in Local Environment

   Before developing your code, you need to set up the development environment according to the [Triton Ascend Installation Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/installation_guide.md).

   To avoid inconsistencies between multiple branches, create a new local development branch for new feature development:

   ```shell
   git checkout -b {new_branch_name} origin/main
   git fetch upstream       # Fetch the latest code from the upstream repository
   git rebase upstream/main # Rebase onto the latest upstream trunk
   ```

   Using the main branch as an example, Triton Ascend may create version branches or downstream development branches as needed. After creating your branch and syncing with the upstream main branch updates, you can start developing your code.

4. Self-Test Code Changes

   After completing code changes, check if your changes pass the tests:

   Write test case code for your developed code under the `ascend/examples/pytest_ut` path of your local branch, and verify your test scripts in the local environment to ensure your changes pass the tests.

5. Push Code to Remote Repository

   After code updates and testing are complete, push your commits to your remote repository.

   ```shell
   git add .
   git status #Check the updated files
   git commit -s -m "your commit message"
   git push origin {your_new_branch_name}
   ```

6. Create a Pull Request to the Triton Ascend Main Repository

   After pushing the code to your remote repository, you need to create a new Pull Request between your new branch and the Triton Ascend main branch. Once the merge request is created, "Jenkins CI" will be automatically set up to build pipeline tests for you. Please merge your Pull Request into the upstream main branch as soon as possible to reduce merge risks.

<h2 id="code-gate-exception-handling">Code Gate Exception Handling</h2>

Code gate exceptions mainly include the following situations. Please resolve the gate exception issues based on the relevant prompt information.

- Compilation Failure

   Based on the prompt information, check the cause of the compilation failure, resolve it, and recompile.

- Static Check Failure

   Based on the prompt information, find the exception information in the code and resolve it.

- CI Pipeline Failure

   Based on the prompt information, find the test cases that caused the CI pipeline failure, check the reasons, resolve them, and rerun the CI pipeline.

<h2 id="issue-specification">ISSUE Specification</h2>

A good way to contribute to the project is to send detailed reports when encountering problems. We always greatly appreciate well-written, thorough bug reports and will be very grateful for them!

When reporting an issue, please refer to the following format:

- What software versions are you using in your environment (Triton Ascend, Python, OS, etc.)?
- Is this a bug report or a feature request?
- What kind of issue are you reporting? Add corresponding labels to highlight it on the issue dashboard.
- What happened?
- What did you expect to happen?
- How can it be reproduced? (Be as precise as possible)

You can also choose one of the predefined [issue filling templates](https://github.com/triton-lang/triton-ascend/issues/new/choose)

Issue Consultation:

- If you find an unresolved issue that you intend to solve, please comment on that issue to let others know you will be responsible for it.
- If an issue has been open for a while, please perform a pre-check before resolving it.
- If you resolve an issue you reported yourself, please let others know before closing the issue.

<h2 id="submitting-a-pr">Submitting a PR</h2>

- Propose your idea as an issue on [GitHub](https://github.com/triton-lang/triton-ascend).
- If the new feature to be developed requires significant design details, you should also submit a design proposal.
- Only after reaching consensus through issue discussion and design proposal review should you proceed with forking, development, and submitting a PR.
- No PR is allowed until it receives 2+ LGTM (Looks Good To Me) from Approvers. Note that reviewers are not allowed to add LGTM to their own PRs.
- After sufficient discussion on the PR, it will be merged, rejected, or abandoned based on the discussion results.

## Notes

- Avoid any irrelevant changes.
- Ensure your commit history is concise and orderly.
- Rebase with the latest upstream repository code before creating a PR.
- For bug fix PRs, ensure all related Issues and PRs are linked.