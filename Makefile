python=python3

clean: ; rm -rf *.reduce.log *.fuzz.log results fuzzing *.recognize.log
clobber: clean;
	-$(MAKE) box-remove
	-rm -rf artifact artifact.tar.gz
	-rm -rf .db
results:; mkdir -p results

find_bugs=07b941b1 93623752 c8491c11 dbcb10e9
grep_bugs=3c3bdace 54d55bba 9c45c193

closure_bugs=2808 2842 2937 3178 3379 1978
clojure_bugs=2092 2345 2450 2473 2518 2521

lua_bugs=5_3_5__4
rhino_bugs=385 386


find_results_src=$(addsuffix .log,$(addprefix results/reduce_find_,$(find_bugs)))
grep_results_src=$(addsuffix .log,$(addprefix results/reduce_grep_,$(grep_bugs)))

lua_results_src=$(addsuffix .log,$(addprefix results/reduce_lua_,$(lua_bugs)))
rhino_results_src=$(addsuffix .log,$(addprefix results/reduce_rhino_,$(rhino_bugs)))
clojure_results_src=$(addsuffix .log,$(addprefix results/reduce_clojure_,$(clojure_bugs)))
closure_results_src=$(addsuffix .log,$(addprefix results/reduce_closure_,$(closure_bugs)))

fuzz_find_results_src=$(addsuffix .log,$(addprefix results/fuzz_find_,$(find_bugs)))
fuzz_grep_results_src=$(addsuffix .log,$(addprefix results/fuzz_grep_,$(grep_bugs)))

recognize_find_results_src=$(addsuffix .log,$(addprefix results/recognize_find_,$(find_bugs)))
recognize_grep_results_src=$(addsuffix .log,$(addprefix results/recognize_grep_,$(grep_bugs)))

recognize_lua_results_src=$(addsuffix .log,$(addprefix results/recognize_lua_,$(lua_bugs)))
recognize_rhino_results_src=$(addsuffix .log,$(addprefix results/recognize_rhino_,$(rhino_bugs)))
recognize_closure_results_src=$(addsuffix .log,$(addprefix results/recognize_closure_,$(closure_bugs)))
recognize_clojure_results_src=$(addsuffix .log,$(addprefix results/recognize_clojure_,$(clojure_bugs)))

BEGIN_CONT=echo CONT=$$CONT;CID=$$(sudo docker ps -a | grep $$CONT | awk '{print $$1}'); echo CID=$$CID; \
  if [ ! -z "$$CID" ]; then \
     CPID=$$(ps -eaf | grep $$CID | grep -v grep | awk '{print $$2}'); echo CPID=$$CPID; \
     if [ ! -z "$$CPID" ]; then

END_CONT=;fi;fi


start_%:; @echo done
stop_%:; @echo done


stop_find: $(addprefix stop_,$(find_bugs))
	@echo done.

stop_grep: $(addprefix stop_,$(grep_bugs))
	@echo done.

$(addprefix start_,$(grep_bugs)):
	-@CONT=$(subst start_,,$@); $(BEGIN_CONT) sudo kill -9 $$CPID $(END_CONT)
	sudo docker stop $(subst start_,,$@)
	sudo docker start $(subst start_,,$@)

$(addprefix stop_,$(grep_bugs)):
	-@CONT=$(subst stop_,,$@); $(BEGIN_CONT) sudo kill -9 $$CPID $(END_CONT)
	sudo docker stop $(subst stop_,,$@)

$(addprefix start_,$(find_bugs)):
	-@CONT=$(subst start_,,$@); $(BEGIN_CONT) sudo kill -9 $$CPID $(END_CONT)
	sudo docker stop $(subst start_,,$@)
	sudo docker start $(subst start_,,$@)

$(addprefix stop_,$(find_bugs)):
	-@CONT=$(subst stop_,,$@); $(BEGIN_CONT) sudo kill -9 $$CPID $(END_CONT)
	sudo docker stop $(subst stop_,,$@)

unbuffer= #unbuffer -p

results/reduce_%.log: src/%.py | results
	@- $(MAKE) start_$(subst find_,,$*)
	@- $(MAKE) start_$(subst grep_,,$*)
	time $(python) $< 2>&1 | $(unbuffer) tee $@_
	@- $(MAKE) stop_$(subst find_,,$*)
	@- $(MAKE) stop_$(subst grep_,,$*)
	mv $@_ $@


results/fuzz_%.log: src/fuzz_%.py results/reduce_%.log
	@- $(MAKE) start_$(subst find_,,$*)
	@- $(MAKE) start_$(subst grep_,,$*)
	time $(python) $< 2>&1 | $(unbuffer) tee $@_
	@- $(MAKE) stop_$(subst find_,,$*)
	@- $(MAKE) stop_$(subst grep_,,$*)
	mv $@_ $@


results/recognize_%.log: src/recognize_%.py results/reduce_%.log
	@- $(MAKE) start_$(subst find_,,$*)
	@- $(MAKE) start_$(subst grep_,,$*)
	time $(python) $< 2>&1 | $(unbuffer) tee $@_
	@- $(MAKE) stop_$(subst find_,,$*)
	@- $(MAKE) stop_$(subst grep_,,$*)
	mv $@_ $@



reduce_find: $(find_results_src); @echo done
reduce_grep: $(grep_results_src); @echo done

reduce_lua: $(lua_results_src); @echo done
reduce_rhino: $(rhino_results_src); @echo done
reduce_clojure: $(clojure_results_src); @echo done
reduce_closure: $(closure_results_src); @echo done

recognize_find: $(recognize_find_results_src); @echo done
recognize_grep: $(recognize_grep_results_src); @echo done

fuzz_find: $(fuzz_find_results_src); @echo done
fuzz_grep: $(fuzz_grep_results_src); @echo done


recognize_lua: $(recognize_lua_results_src); @echo done
recognize_rhino: $(recognize_rhino_results_src); @echo done
recognize_clojure: $(recognize_clojure_results_src); @echo done
recognize_closure: $(recognize_closure_results_src); @echo done


all_find: recognize_find fuzz_find
	tar -cf find.tar results .db
	@echo find done

all_grep: recognize_grep fuzz_grep
	tar -cf grep.tar results .db
	@echo grep done

all_lua: recognize_lua
	tar -cf lua.tar results .db
	@echo lua done

all_rhino: recognize_rhino
	tar -cf rhino.tar results .db
	@echo rhino done

all_clojure: recognize_clojure
	tar -cf clojure.tar results .db
	@echo clojure done

all_closure: recognize_closure
	tar -cf closure.tar results .db
	@echo closure done

all: all_grep all_find all_lua all_rhino all_clojure all_closure
	@echo done

dbgbench-init: .dbgbench init-find init-grep
	@echo done

.dbgbench:
	git clone https://github.com/vrthra-forks/dbgbench.github.io.git
	cat dbgbench.patch | (cd dbgbench.github.io/; patch -p1)
	touch $@

dbgbench-clobber:
	-$(MAKE) rm-find
	-$(MAKE) rm-grep
	rm -rf dbgbench.github.io .dbgbench

init-find: .dbgbench;
	for i in $(find_bugs); do \
		$(MAKE) -C dbgbench.github.io/docker initfind-$$i; \
		sudo docker stop $$i; \
		done
init-grep: .dbgbench;
	for i in $(grep_bugs); do \
		$(MAKE) -C dbgbench.github.io/docker initgrep-$$i; \
		sudo docker stop $$i; \
		done

rm-find:; $(MAKE) -C dbgbench.github.io/docker rm-find
rm-grep:; $(MAKE) -C dbgbench.github.io/docker rm-grep

prune-find:; sudo docker system prune --filter ancestor=falgebra/find || echo
prune-grep:; sudo docker system prune --filter ancestor=falgebra/grep || echo

ls-find:; @sudo docker ps -a --filter ancestor=falgebra/find --format 'table {{.Image}} {{.ID}} {{.Names}} {{.Status}}'
ls-grep:; @sudo docker ps -a --filter ancestor=falgebra/grep --format 'table {{.Image}} {{.ID}} {{.Names}} {{.Status}}'

artifact.tar.gz: Vagrantfile Makefile
	rm -rf artifact && mkdir -p artifact/falgebra
	cp README.md artifact/README.txt
	cp -r README.md lang src dbgbench.github.io .dbgbench Makefile Vagrantfile etc/jupyter_notebook_config.py artifact/falgebra
	cp -r Vagrantfile artifact/
	tar -cf artifact1.tar artifact
	gzip artifact1.tar
	mv artifact1.tar.gz artifact.tar.gz



# PACKAGING
box-create: falgebra.box
falgebra.box: artifact.tar.gz
	cd artifact && vagrant up
	cd artifact && vagrant ssh -c 'cd /vagrant; tar -cpf ~/falgebra.tar falgebra ; cd ~/; tar -xpf ~/falgebra.tar; rm -f ~/falgebra.tar'
	cd artifact && vagrant ssh -c 'mkdir -p /home/vagrant/.jupyter; cp /vagrant/falgebra/jupyter_notebook_config.py /home/vagrant/.jupyter/jupyter_notebook_config.py'
	cd artifact && vagrant ssh -c 'cd ~/falgebra && make dbgbench-init' # TODO: for final
	cd artifact && vagrant package --output ../falgebra1.box --vagrantfile ../Vagrantfile.new
	mv falgebra1.box falgebra.box

box-hash:
	md5sum falgebra.box

box-add: falgebra.box
	-vagrant destroy $$(vagrant global-status | grep falgebra | sed -e 's# .*##g')
	rm -rf vtest && mkdir -p vtest && cp falgebra.box vtest
	cd vtest && vagrant box add falgebra ./falgebra.box
	cd vtest && vagrant init falgebra
	cd vtest && vagrant up

box-status:
	-vagrant global-status | grep falgebra
	-vagrant box list | grep falgebra

box-remove1:
	cd artifact; vagrant destroy

box-remove2:
	cd vtest; vagrant destroy

box-remove:
	-vagrant destroy $$(vagrant global-status | grep falgebra | sed -e 's# .*##g')
	vagrant box remove falgebra

show-ports:
	 sudo netstat -ln --program | grep 8888

box-up1:
	cd artifact; vagrant up

box-up2:
	cd vtest; vagrant up

box-connect1:
	cd artifact; vagrant up; vagrant ssh
box-connect2:
	cd vtest; vagrant ssh


VM=

vm-list:
	VBoxManage list vms

vm-remove:
	-VBoxManage startvm $(VM)  --type emergencystop
	-VBoxManage controlvm $(VM)  poweroff
	VBoxManage unregistervm $(VM) -delete
