document.addEventListener('DOMContentLoaded',()=>{
  // simple parallax shimmer on hero-left overlay
  const overlay = document.querySelector('.overlay-card');
  if(overlay){
    overlay.addEventListener('mousemove', (e)=>{
      const r = overlay.getBoundingClientRect();
      const x = ((e.clientX - r.left)/r.width - .5)*6;
      const y = ((e.clientY - r.top)/r.height - .5)*-6;
      overlay.style.transform = `translateY(${y}px) rotateX(${y/4}deg) rotateY(${x/4}deg)`;
    });
    overlay.addEventListener('mouseleave', ()=> overlay.style.transform='');
  }
});
