const footer = document.querySelector("[data-shared-footer]");

if (footer) {
  footer.innerHTML = `
    <div class="footer-shell">
      <div class="footer-main">
        <div class="footer-brand">
          <a class="footer-logo" href="https://www.apodex.com/" aria-label="Apodex"><img src="mark-footer.svg?v=2" alt="" /><span>Apodex</span></a>
          <div class="footer-brand-links"><a href="https://www.apodex.com/blog/apodex">About Us</a><a href="mailto:support@apodex.com">Contact Us <span aria-hidden="true">✉</span></a></div>
        </div>
        <div class="footer-nav">
          <div class="footer-column"><h2>Solutions</h2><a href="https://www.apodex.ai/" target="_blank" rel="noreferrer">Apodex AI ↗</a><a href="https://platform.apodex.ai/" target="_blank" rel="noreferrer">Apodex API ↗</a><a href="https://github.com/ApodexAI" target="_blank" rel="noreferrer">Github ↗</a><a href="https://huggingface.co/apodex" target="_blank" rel="noreferrer">HuggingFace ↗</a></div>
          <div class="footer-column"><h2>Models</h2><a href="https://www.apodex.com/research">Deep Research</a><a href="https://www.apodex.com/solve">Deep Solve</a><a href="https://www.apodex.com/discover">Deep Discover</a></div>
          <div class="footer-column"><h2>Community</h2><a href="https://discord.com/invite/TDJA59TCng" target="_blank" rel="noreferrer">Discord ↗</a><a href="https://x.com/Apodex_AI" target="_blank" rel="noreferrer">X.com ↗</a></div>
          <div class="footer-column"><h2>Terms &amp; Policies</h2><a href="https://www.apodex.com/policies/privacy">Privacy Policy</a><a href="https://www.apodex.com/policies">Other Policies</a></div>
        </div>
      </div>
      <div class="footer-bottom">
        <div class="footer-copyright"><span>© 2026 Apodex. All rights reserved.</span></div>
        <div class="footer-socials" aria-label="Apodex social links">
          <a href="https://x.com/Apodex_AI" target="_blank" rel="noreferrer" aria-label="X"><svg aria-hidden="true"><use href="footer-socials.svg#x"></use></svg></a>
          <a href="https://www.linkedin.com/company/114344449/admin/dashboard/" target="_blank" rel="noreferrer" aria-label="LinkedIn"><svg aria-hidden="true"><use href="footer-socials.svg#linkedin"></use></svg></a>
          <a href="https://discord.com/invite/TDJA59TCng" target="_blank" rel="noreferrer" aria-label="Discord"><svg aria-hidden="true"><use href="footer-socials.svg#discord"></use></svg></a>
          <a href="https://github.com/ApodexAI" target="_blank" rel="noreferrer" aria-label="GitHub"><svg aria-hidden="true"><use href="footer-socials.svg#github"></use></svg></a>
        </div>
      </div>
    </div>`;
}
