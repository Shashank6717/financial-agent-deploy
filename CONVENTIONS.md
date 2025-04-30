# Engineering Conventions

## Conventions
> When writing code, follow these conventions.

- Write simple verbose code over terse, compact dense code.
- If a function does not have a corresponding test, then mention it.
- When building tests, avoid using mocks when possible. 
- Tests must go in a top level tests folder, with a folder structure that shadows the project structure
- Test files should follow the pre-pended 'test_' naming convention

## Project Structure

- `metropolis/` - Project Source Code
- `metrolpolis-deployment/` - Helm Charts
- `tests/` - Tests
