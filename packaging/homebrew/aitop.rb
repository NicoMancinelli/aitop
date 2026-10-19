class Aitop < Formula
  include Language::Python::Virtualenv

  desc "AI-centric neofetch meets btop for local AI runtimes"
  homepage "https://github.com/NicoMancinelli/aitop"
  url "https://github.com/NicoMancinelli/aitop/archive/refs/tags/v0.3.0.tar.gz"
  sha256 "REPLACE_WITH_V0.3.0_SHA256"
  license "MIT"
  head "https://github.com/NicoMancinelli/aitop.git", branch: "main"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/aitop", "--version"
    system "#{bin}/aitop", "doctor"
  end
end
