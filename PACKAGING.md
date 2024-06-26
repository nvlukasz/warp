# Release Instructions

## Versioning

Versions take the format X.Y.Z, similar to [Python itself](https://devguide.python.org/developer-workflow/development-cycle/#devcycle):

- Increments in X are reserved for major reworks of the project causing disruptive incompatibility (or reaching the 1.0 milestone).
- Increments in Y are for regular releases with a new set of features.
- Increments in Z are for bug fixes. In principle there are no new features. Can be omitted if 0 or not relevant.

This is similar to [Semantic Versioning](https://semver.org/) but less strict around backward compatibility.
Like with Python, some breaking changes can be present between minor versions if well documented and gradually introduced.

Note that prior to 0.11.0 this schema was not strictly adhered to.

## Repositories

Development happens internally on a GitLab repository (part of the Omniverse group), while releases are made public on GitHub.

This document uses the following Git remote names:

- **omniverse**: `git remote add omniverse https://gitlab-master.nvidia.com/omniverse/warp.git`
- **github**: `git remote add github https://github.com/NVIDIA/warp.git`

Currently, all feature branches get merged into the `main` branch of the **omniverse** repo and then GitLab push-mirrors
the changes over to GitHub (nominally within five minutes). This mirroring process also pushes all tags
(only tags beginning with `v` are allowed to be created) and branches beginning with `release-`.

The status of push mirroring can be checked under **Settings** :arrow_right: **Repository** on GitLab.

## GitLab Release Branch

1) Create a branch in your fork repository from which a merge-request will be opened to bump the version string
   and create the public-facing changelogs for the release.

2) Search & replace the current version string from `VERSION.md`.

   We want to keep the Omniverse extensions' version in sync with the library so update the strings found in the `exts` folder as well.

   The version string currently appears in the following two files, but there could be more in the future:

   - `omni.warp/config/extension.toml`
   - `omni.warp.core/config/extension.toml`

   Be sure *not* to update previous strings in `CHANGELOG.md`.

3) Update `CHANGELOG.md` from Git history (since the last release branch). Only list user-facing changes.

   The entire development team should all be helping to keep this file up-to-date, so verify that all changes users
   should know about are included.

   The changelogs from the Omniverse extensions found in `exts` are kept in sync with the one from the library, so update them all at the same time and list any change made to the extensions.

4) Open a MR on GitLab to merge this branch into `main`. Send a message in `#omni-warp-dev` to the `@warp-team`
   asking for a review of the merge request's changes.

5) Merge the branch into `main` after waiting a reasonable amount of time for the team to review and approve the MR.

6) For new `X.Y` versions, create a release branch (note `.Z` maintenance versions remain on the same branch):

   `git checkout -b release-X.Y [<start-point>]`

   If branching from an older revision or reusing a branch, make sure to cherry-pick the version and changelog update.

7) Make any release-specific changes (e.g. disable/remove features not ready yet).

8) :warning: Keep in mind that branches pushed to the **omniverse** repository beginning with `release-` are
   automatically mirrored to GitLab. :warning:

   Push the new release branch to **omniverse** when it is in a state ready for CI testing.

9) Check that the last revision on the release branch passes GitLab CI tests. A pipeline should have been automatically
   created after pushing the branch in the previous step:

   <https://gitlab-master.nvidia.com/omniverse/warp/-/pipelines>

   Fix issues until all tests pass. Cherry-pick fixes for `main` where applicable.

## Creating a GitHub Release Package

1) Wait for the (latest) packages to appear in:

   <https://gitlab-master.nvidia.com/omniverse/warp/-/packages/>

2) Download the `.whl` files for each supported platform and move them into an empty folder.

3) Run tests for at least one platform:

    - Run `python -m pip install warp_lang-<version>-<platform-tag>.whl`
    - Run `python -m warp.tests`

    Check that the correct version number gets printed.

4) If tests fail, make fixes on `release-X.Y` and where necessary cherry-pick to `main` before repeating from step (1).

5) Tag the release with `vX.Y.Z` on `release-X.Y` and push to `omniverse`.
   Both the tag and the release branch will be automatically mirrored to GitLab.

   It is safest to push *just* the new tag using `git push omniverse vX.Y.Z`.

   In case of a mistake, a tag already pushed to `omniverse` can be deleted from the GitLab UI.
   The bad tag must also be deleted from the GitHub UI if it was mirrored there.

6) Create a new release on [GitHub](https://github.com/NVIDIA/warp) with a tag and title of `vX.Y.Z` and
   upload the `.whl` artifacts as attachments. Use the changelog updates as the description.

## Upload a PyPI Release

First time:

- Create a [PyPI](https://pypi.org/) account.
- [Create a Token](https://pypi.org/manage/account/#api-tokens) for uploading to the `warp-lang` project (store it somewhere safe).
- Get an admin (<mmacklin@nvidia.com>) to give you write access to the project.

Per release:

Remove any `.whl` files from the upload folder that contain a `+cpu` or `+cu` (local) tag.

Run `python -m twine upload *` from the `.whl` packages folder (on Windows make sure to use `cmd` shell; Git Bash doesn't work).

- username: `__token__`
- password: `(your token string from PyPI)`

## Publishing the Omniverse Extensions

1) Ensure that the version strings and `CHANGELOG.md` files in the `exts` folder are in sync with the ones from the library.

2) Wait for the (latest) packages to appear in:

   <https://gitlab-master.nvidia.com/omniverse/warp/-/packages/>

3) Download `kit-extensions.zip` to your computer.

4) Extract it to a clean folder and check the extensions inside of Kit:

    - Run `omni.create.sh --ext-folder /path/to/artifacts/exts --enable omni.warp-X.Y.Z --enable omni.warp.core-X.Y.Z`
    - Ensure that the example scenes are working as expected
    - Run test suites for both extensions

5) If tests fail, make fixes on `release-X.Y` and where necessary cherry-pick to `main` before repeating from step (2).

6) If all tests passed:

   - `kit --ext-folder /path/to/artifacts/exts --publish omni-warp.core-X.Y.Z`
   - `kit --ext-folder /path/to/artifacts/exts --publish omni-warp-X.Y.Z`

7) Ensure that the release is tagged with `vX.Y.Z` on both `omniverse/release-X.Y` and `github/release-X.Y`.

## Automated processes

The following is just for your information. These steps should run automatically by CI/CD pipelines, but can be replicated manually if needed:

### Building the documentation

The contents of <https://nvidia.github.io/warp/> is generated by a GitHub pipeline which runs `python build_docs.py` (prerequisites: `pip install docs/requirements.txt`).

### Building pip wheels

The GitLab pipeline's `create pypi wheels` Job (part of the `package` Stage) combines artifacts from each platform build, moving the contents of `warp/bin` to platform- and architecture-specific
subfolders; e.g. `warp/bin/linux-x86_64` and `warp/bin/linux-aarch64` both contain `warp.so` and `warp-clang.so` files.

Pip wheels are then built using:

```bash
python -m build --wheel -C--build-option=-Pwindows-x86_64
python -m build --wheel -C--build-option=-Plinux-x86_64
python -m build --wheel -C--build-option=-Plinux-aarch64
python -m build --wheel -C--build-option=-Pmacos-universal
```

Selecting the correct library files for each wheel happens in [`setup.py`](setup.py).
