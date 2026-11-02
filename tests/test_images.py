import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import images
class ImageTests(unittest.TestCase):
    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):images.resolve_image('cpu-sci',version='never',platform='linux/amd64')
    def test_unpublished_image_never_scores(self):
        config={'namespace':'registry.example/abench','images':{}}
        with patch.object(images,'manifest',return_value=config), patch.object(images,'lock_entries',return_value={}):
            with self.assertRaises(ValueError):images.resolve_image('cpu-sci',platform='linux/amd64')
            with patch.dict(os.environ,{'ABENCH_IMAGE_CPU_SCI':'local:test'}):
                with self.assertRaises(ValueError):images.resolve_image('cpu-sci',platform='linux/amd64')
                self.assertEqual(images.resolve_image('cpu-sci',platform='linux/amd64',allow_unlocked=True),'local:test')
    def test_profile_parent_preserves_python(self):
        p=images.profiles()
        self.assertEqual(len(p),7)
        for name,item in p.items():
            self.assertEqual(item['version'],'1.0.0')
            for version in item['packages'].values():self.assertNotIn('>',version)
        text=(images.ROOT/'docker/profiles/gpu-torch-hf/Dockerfile').read_text()
        self.assertIn('FROM ghcr.io/huicir/abench/gpu-torch:1.0.0',text)
        self.assertIn('COPY --from=cuda /usr/local/cuda',(images.ROOT/'docker/profiles/gpu-torch-cuda-dev/Dockerfile').read_text())
        gpu=(images.ROOT/'docker/profiles/gpu-torch/Dockerfile').read_text()
        self.assertRegex(gpu,r'install -y --no-install-recommends procps .*poppler-utils gcc g\+\+')
        self.assertIn('poppler-utils',(images.ROOT/'docker/profiles/cpu-sci/Dockerfile').read_text())
        self.assertNotIn('poppler-utils',text)                # children inherit the root profile's tools
    def test_versions_are_semantic_versions_not_dates(self):
        import re
        semver=re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')
        for name in images.profiles():
            self.assertRegex(images.image_version(name),semver,name)
        for name,image in images.manifest()['images'].items():
            if 'version' in image:self.assertRegex(str(image['version']),semver,name)
    def test_versions_are_locked_side_by_side(self):
        old='sha256:'+'1'*64;new='sha256:'+'2'*64;agent='sha256:'+'3'*64
        lock={'cpu-sci@0.9.0':{'linux/amd64':{'digest':old,'reference':'ghcr.io/huicir/abench/cpu-sci@'+old}},
              'cpu-sci@1.0.0':{'linux/amd64':{'digest':new,'reference':'ghcr.io/huicir/abench/cpu-sci@'+new}}}
        with patch.object(images,'lock_entries',return_value=lock):
            self.assertTrue(images.resolve_image({'profile':'cpu-sci','version':'0.9.0'},platform='linux/amd64').endswith(old))
            self.assertTrue(images.resolve_image('cpu-sci',platform='linux/amd64').endswith(new))
            # every agent runs in the task's profile image itself; its CLI comes from a harness bundle
            self.assertEqual(images.resolve_image('cpu-sci@1.0.0','claude',platform='linux/amd64'),
                             images.resolve_image('cpu-sci@1.0.0',platform='linux/amd64'))
            self.assertEqual(images.known_versions('cpu-sci'),{'0.9.0','1.0.0'})
        with patch.object(images,'lock_entries',return_value={}):
            self.assertEqual(images.resolve_image('cpu-sci','claude',platform='linux/amd64',allow_unlocked=True),
                             'ghcr.io/huicir/abench/cpu-sci:'+images.image_version('cpu-sci'))
            with self.assertRaisesRegex(ValueError,'unknown profile version'):
                images.resolve_image({'profile':'cpu-sci','version':'0.9.0'},platform='linux/amd64',allow_unlocked=True)
        self.assertEqual(images.env_lock_path('cpu-sci','1.0.0','linux/arm64').name,'cpu-sci@1.0.0-linux-arm64.json')
        for name in images.profiles():
            if name in ('cpu-sci','gpu-torch'):
                self.assertTrue(images.env_lock_path(name,images.image_version(name)).is_file(),name)
    def test_lock_is_immutable_per_version_and_new_versions_add_entries(self):
        import shutil,tempfile
        digest=lambda c:'sha256:'+c*64
        class FakeDocker:
            def __init__(self,ref):self.ref=ref
            def info(self):return {'platform':'linux/amd64'}
            def inspect(self,tag):return {'RepoDigests':[self.ref]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'spec/env').mkdir(parents=True)
            shutil.copyfile(images.ROOT/'spec/env/images.yaml',root/'spec/env/images.yaml')
            shutil.copyfile(images.ROOT/'spec/env/profiles.yaml',root/'spec/env/profiles.yaml')
            (root/'spec/env/images.lock.yaml').write_text('schema: abench.images_lock/v1\nimages: {}\n')
            with patch.object(images,'ROOT',root):
                ref='ghcr.io/huicir/abench/cpu-sci@'+digest('a')
                images.manage('lock','cpu-sci',docker=FakeDocker(ref))
                with self.assertRaisesRegex(ValueError,'immutable'):
                    images.manage('lock','cpu-sci',docker=FakeDocker('ghcr.io/huicir/abench/cpu-sci@'+digest('b')))
                text=(root/'spec/env/profiles.yaml').read_text().replace("version: 1.0.0","version: 1.1.0",1)
                (root/'spec/env/profiles.yaml').write_text(text)
                images.manage('lock','cpu-sci',docker=FakeDocker('ghcr.io/huicir/abench/cpu-sci@'+digest('b')))
                self.assertEqual(set(images.lock_entries()),{'cpu-sci@1.0.0','cpu-sci@1.1.0'})
                self.assertEqual(images.known_versions('cpu-sci'),{'1.0.0','1.1.0'})
    def test_local_lock_scores_only_the_exact_image_id(self):
        import shutil,tempfile
        class FakeDocker:
            ids={}
            def __init__(self,*a,**k):pass
            def info(self):return {'platform':'linux/amd64','resources_enforced':True}
            def inspect(self,ref):return {'Id':self.ids.get(ref,'sha256:'+'0'*64),'RepoDigests':[]}
        tag='ghcr.io/huicir/abench/cpu-sci:'+images.image_version('cpu-sci')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'spec/env').mkdir(parents=True)
            for name in ('images.yaml','profiles.yaml'):shutil.copyfile(images.ROOT/'spec/env'/name,root/'spec/env'/name)
            (root/'spec/env/images.lock.yaml').write_text('schema: abench.images_lock/v1\nimages: {}\n')
            with patch.object(images,'ROOT',root),patch.object(images,'Docker',FakeDocker):
                FakeDocker.ids={tag:'sha256:'+'a'*64}
                self.assertEqual(images.manage('lock','cpu-sci',docker=FakeDocker(),local=True)['locked_local'],'sha256:'+'a'*64)
                attested=images.image_attestation(tag,{'profile':'cpu-sci','version':'1.0.0'})
                self.assertEqual((attested['locked'],attested['lock']),(True,'local'))
                self.assertEqual(images.manage('verify','cpu-sci',docker=FakeDocker())['lock'],'local')
                FakeDocker.ids={tag:'sha256:'+'b'*64}   # rebuilt image under the same tag
                attested=images.image_attestation(tag,{'profile':'cpu-sci','version':'1.0.0'})
                self.assertEqual((attested['locked'],attested['lock']),(False,None))
                with self.assertRaisesRegex(ValueError,'lock it again'):
                    images.manage('verify','cpu-sci',docker=FakeDocker())
                import catalog
                with patch.object(catalog,'ROOT',images.ROOT):
                    self.assertEqual(catalog.case_lock('DS-PIPE-002')['judge_image_lock']['linux/amd64']['image_id'],'sha256:'+'a'*64)
                import judge_container
                import yaml
                oj=yaml.safe_load((Path(__file__).resolve().parents[1]/'cases/DS-PIPE-002/case/oj.yaml').read_text())
                with patch.object(judge_container,'Docker',FakeDocker):
                    self.assertEqual(judge_container.DockerStageLauncher.for_case(oj).lock,None)
                    FakeDocker.ids={tag:'sha256:'+'a'*64}
                    launcher=judge_container.DockerStageLauncher.for_case(oj)
                    self.assertEqual((launcher.locked,launcher.lock),(True,'local'))
                digest='sha256:'+'c'*64
                (root/'spec/env/images.lock.yaml').write_text(
                    'images:\n  cpu-sci@1.0.0:\n    linux/amd64: {digest: %s, reference: ghcr.io/huicir/abench/cpu-sci@%s}\n'%(digest,digest))
                with self.assertRaisesRegex(ValueError,'is published'):
                    images.manage('lock','cpu-sci',docker=FakeDocker(),local=True)
    def test_local_locks_are_noted_in_the_score_record(self):
        from agents.scoring import apply_agent_attestation
        evaluation={'scorable':True,'judge_dependency':{'image_lock':'local'}}
        record={'run_mode':'benchmark','isolation':'container','harness':'mock',
                'sandbox':{'backend':'docker','verified':True,'image_attestation':{'locked':True,'lock':'local'},
                           'resources_enforced':True,'hardware_matches':True,'platform':'linux/amd64'},
                'network':{'mode':'none'},'budget':{'over_budget':False},'status':'succeeded'}
        apply_agent_attestation(evaluation,record)
        self.assertEqual(evaluation['image_locks'],{'agent':'local','judge':'local','sidecar':None,'harness':None})
        self.assertIn('unpublished',evaluation['image_lock_note'])
if __name__=='__main__':unittest.main()
