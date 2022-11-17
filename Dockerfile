FROM ajevnisek/tau-base-docker:latest

COPY environment.yml .
RUN conda update -y conda && \
    conda env update --file environment.yml --name base

WORKDIR /storage/jevnisek

# allows ssh to the docker from outside the container
RUN sed 's@.*AuthorizedKeysFile.*@AuthorizedKeysFile /storage/jevnisek/.ssh/authorized_keys@' -i /etc/ssh/sshd_config
RUN chmod 0700 /storage/jevnisek

# allows the docker to access the outside world:
RUN sed 's@.*IdentityFile.*@IdentityFile /storage/jevnisek/.ssh/id_ed25519@' -i /etc/ssh/ssh_config
RUN sed 's@.*StrictHostKeyChecking.*@StrictHostKeyChecking off@' -i /etc/ssh/ssh_config

# do not ask for git password
RUN sed 's@.*BatchMode.*@BatchMode yes@' -i /etc/ssh/ssh_config

WORKDIR /
## clone repo and cd to a directory where the main is:
COPY clone_repo_and_run_python_scripts.sh .
ENTRYPOINT ["bash", "clone_repo_and_run_python_scripts.sh"]