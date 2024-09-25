# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['MuseGUI_Student.py'],
             pathex=['C:\\Users\\dafne\\Desktop\\museGUI', 'C:\\Users\\dafne\\Desktop\\museGUI\\MuseGUI2\\Lib\\site-packages', 'C:\\Users\\dafne\\Desktop\\museGUI\\MuseGUI2\\lib\\site-packages\\cv2'],
             binaries=[],
             datas=[('C:\\Users\\dafne\\Desktop\\museGUI\\MuseGUI2\\Lib\\site-packages\\pylsl\\*', 'pylsl\\lib'),
             ('icons\\*', 'icons'),
             ('C:\\Users\\dafne\\Desktop\\museGUI\\MuseGUI2\\Lib\\site-packages\\google_api_python_client-*', 'google_api_python_client-1.12.8.dist-info'),
             ('C:\\Users\\dafne\\Desktop\\museGUI\\DriveAPI', 'DriveAPI')],
             hiddenimports=['sklearn', 'googleapiclient', 'apiclient'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='MuseGUI_Student',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False)
