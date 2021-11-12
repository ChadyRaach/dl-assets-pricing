
# Getting started

Change directory (`cd`) into the repository, then type:

```bash
# cd dl-assets-pricing
conda env create -f environment.yml
source activate my-env
```
## Utilities

You can add these aliases to your global git config file.
````
git config --global core.editor
````
Then add these lines:
````
[alias]
    co = checkout
    ll = log --pretty=format:"%C(yellow)%h%Cred%d\\ %Creset%s%Cblue\\ [%cn]" --decorate --numstat
    st = status
    ci = commit
````
## Guidelines
To contribute in this repository, it is recommended to follow these steps:
1. Create a new branch locally and remotely
2. Add your own code
3. Push your code to the remote branch and create a pull request
4. Ask for someone from the team to review your pull request
5. Once the PR is reviewed, check that there are no merge conflicts, merge your branch and DELETE it.
